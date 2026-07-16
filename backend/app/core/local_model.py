"""本地模型解析：把 HuggingFace model_id 解析成可用的本地目录。

加载顺序：
  1) 工作目录下 .model/<model_id>/ 已经就绪（有 config.json）→ 直接用
  2) HF 默认 cache 里有同名快照 → 复制快照到 .model/<model_id>/，再用本地副本
  3) 都没有 → snapshot_download 直接下到 .model/<model_id>/（不走 HF 默认 cache）

设计目的：
- 项目级别的模型副本（.model/）让不同机器 / 不同容器里启动都能稳定走同一份
  权重，不会因为 HF cache 被删/被破坏而失败。
- 第二步保留 HF cache 兼容：用户已经手动 hf download 过的模型不必再下载一遍。
- 第三步不走 HF 默认 cache，避免 cache 体积膨胀 + 不被 .model/ 接管。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

# 模块级缓存：避免每次 resolve_local_path 都重复写 os.environ。
_hf_endpoint_applied: Optional[str] = None


def _apply_hf_endpoint(endpoint: str | None) -> None:
    """把 HF_ENDPOINT 写到进程环境里，huggingface_hub / sentence_transformers 会读这个。

    pydantic-settings 不会把 .env 里未在 Settings 中声明的字段写回 os.environ，
    所以镜像配置需要这里显式 export。模块级缓存避免重复覆盖。
    """
    global _hf_endpoint_applied
    if not endpoint:
        return
    if _hf_endpoint_applied == endpoint:
        return
    os.environ["HF_ENDPOINT"] = endpoint
    _hf_endpoint_applied = endpoint
    logger.info("[local-model] HF_ENDPOINT applied: {}", endpoint)


def _has_weights(path: Path) -> bool:
    """检查目录是否含模型权重文件。仅 config.json 不算完整（BGE-M3 等经常
    下载 config + tokenizer 成功但主权重失败/中断，留下半成品 cache）。"""
    candidates = ("model.safetensors", "pytorch_model.bin", "model.bin")
    return any((path / f).is_file() for f in candidates)


def _find_hf_cached_snapshot(model_id: str) -> Optional[str]:
    """在 HF 默认 cache 目录下找 model_id 的 snapshot 路径，返回第一个含
    config.json + 权重文件 的完整快照。

    HF 缓存目录布局：
        {HF_HOME or ~/.cache/huggingface}/hub/models--{org}--{name}/snapshots/<sha>/
    """
    hf_home = Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface" / "hub")
    snapshots_dir = hf_home / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    snapshots = sorted(d for d in snapshots_dir.iterdir() if d.is_dir())
    for snap in snapshots:
        if (snap / "config.json").is_file() and _has_weights(snap):
            return str(snap)
    return None


def _ensure_local_model(model_id: str, local_dir: str) -> str:
    """确保模型在 local_dir 可用，逻辑见模块 docstring。返回可用本地目录。

    自愈：
    - 本地目录有 config 但缺权重 → 当作半成品，先清掉再走 cache / 重新下载路径
    - HF cache 也是半成品 → 跳过 cache，直接走 snapshot_download
    """
    local_path = Path(local_dir)
    if local_path.is_dir() and (local_path / "config.json").is_file() and _has_weights(local_path):
        logger.info("[local-model] 本地目录已就绪: {}", local_dir)
        return local_dir
    if local_path.is_dir():
        # 半成品，留着没用；显式清掉，让后续步骤（cache 复制 / 下载）从干净环境开始
        logger.warning("[local-model] 本地目录不完整（缺 config 或权重），清掉重来: {}", local_dir)
        shutil.rmtree(local_path)

    cached = _find_hf_cached_snapshot(model_id)
    if cached:
        logger.info("[local-model] 从 HF cache 复制到本地: {} → {}", cached, local_dir)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cached, local_dir)
        return local_dir

    logger.info("[local-model] 本地/HF cache 都缺失或不完整，下载到本地: {} → {}", model_id, local_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # 延迟导入：避免模块加载阶段硬依赖 huggingface_hub
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id, local_dir=local_dir)
    return local_dir


def resolve_local_path(
    model_id: str,
    *,
    root: Path | str | None = None,
    override: str | None = None,
    hf_endpoint: str | None = None,
) -> str:
    """把 model_id 解析成本地可用路径。

    Args:
        model_id: HuggingFace 模型 id（如 ``BAAI/bge-m3``）。
        root: 本地模型根目录；默认 ``cwd/.model``。
        override: 显式本地目录覆盖；非空时优先用该路径（不存在则下载到该路径）。
        hf_endpoint: HuggingFace 镜像 URL（如 https://hf-mirror.com）；留空走默认。

    Returns:
        可直接传给 ``from_pretrained`` 的本地目录字符串。
    """
    if hf_endpoint:
        _apply_hf_endpoint(hf_endpoint)

    if override:
        ovr = Path(override)
        if ovr.is_dir() and (ovr / "config.json").is_file() and _has_weights(ovr):
            logger.info("[local-model] override 命中本地: {}", override)
            return override
        if ovr.is_dir():
            logger.warning("[local-model] override 目录不完整，清掉重来: {}", override)
            shutil.rmtree(ovr)
        logger.info("[local-model] override 指定路径不存在，将下载: {}", override)
        return _ensure_local_model(model_id, override)

    root_path = Path(root) if root else Path.cwd() / ".model"
    local_dir = str(root_path / model_id)
    return _ensure_local_model(model_id, local_dir)
