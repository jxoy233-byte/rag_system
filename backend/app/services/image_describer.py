"""图片描述服务：调用 VLM 为图片生成文字描述。

支持两种模式：
- OpenAI API（gpt-4o-mini 等）
- 本地 VLM（Qwen3-VL-2B-Instruct）

本地 VLM 按需加载，处理完后可释放内存。
"""

from __future__ import annotations

import base64
import io
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from app.loaders.base import ImageInfo
from PIL import Image as PILImage  # 模块级 import：_compress_image 等子函数也能用


# 图片描述统一提示词：宽松版。
# 用户希望多说点但有上限，配合 max_new_tokens=600，模型自然停在 1-3 句。
# 否定指引只挡最耗时的「无意义展开」式啰嗦，不限制真正的内容。
IMAGE_DESCRIBE_PROMPT = (
    "用 1-3 句中文客观描述这张图片：图类型（图/表/示意/截图/公式等）、"
    "图里可见的关键文字或数据、图要表达的核心观点。"
    "信息密度高一点但不要重复、不要客套、不要「这张图显示了」开头。"
)


# ===== 描述器抽象 =====


class ImageDescriber(ABC):
    """图片描述器接口。"""

    @abstractmethod
    async def describe(self, images: list[ImageInfo]) -> list[ImageInfo]:
        """批量生成图片描述。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放资源（如卸载模型）。"""
        ...


# ===== OpenAI 实现 =====


class OpenAIImageDescriber(ImageDescriber):
    """使用 OpenAI API（gpt-4o-mini）生成图片描述。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        max_tokens: int = 600,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens

    async def describe(self, images: list[ImageInfo]) -> list[ImageInfo]:
        if not images:
            return images

        import httpx

        for img in images:
            try:
                # 转为 base64
                b64 = base64.b64encode(img.image_bytes).decode("utf-8")
                data_url = f"data:{img.mime_type};base64,{b64}"

                # 调用 OpenAI Vision API
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": IMAGE_DESCRIBE_PROMPT,
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": data_url},
                                        },
                                    ],
                                }
                            ],
                            "max_tokens": self.max_tokens,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    img.description = data["choices"][0]["message"]["content"].strip()
                    logger.debug(
                        "OpenAI image described: {} -> {} chars",
                        img.mime_type,
                        len(img.description),
                    )
            except Exception as e:
                logger.warning("OpenAI image description failed: {}", e)
                img.description = "[图片描述生成失败]"

        return images

    def close(self) -> None:
        pass  # 无资源需要释放


# ===== 本地 VLM 实现 =====


class LocalVLMDescriber(ImageDescriber):
    """使用本地 Qwen3-VL-2B-Instruct 生成图片描述。

    按需加载模型，处理完成后可调用 close() 释放内存。
    """

    _instance: Optional["LocalVLMDescriber"] = None
    _model = None
    _processor = None

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        local_dir: Optional[str] = None,
        device: str = "auto",
        max_pixels: int = 256 * 28 * 28,  # ≈ 256x256 — 再低会让图表里的字看不清；再高会拖慢入库 2-3 倍
    ) -> None:
        self.model_id = model_id
        self.local_dir = local_dir or str(Path.cwd() / ".model" / model_id.replace("/", "--"))
        self.device = device
        self.max_pixels = max_pixels
        self._loaded = False

    def _ensure_model(self) -> str:
        """确保模型可用，返回本地路径。"""
        local_path = Path(self.local_dir)

        # 1. 本地目录已有完整模型
        if local_path.is_dir() and (local_path / "config.json").is_file():
            logger.info("[VLM] 本地目录已就绪: {}", self.local_dir)
            return self.local_dir

        # 2. 从 HF cache 复制
        cached = self._find_hf_cached_snapshot(self.model_id)
        if cached:
            logger.info("[VLM] 从 HF cache 搬到本地: {} → {}", cached, self.local_dir)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cached, self.local_dir)
            return self.local_dir

        # 3. 直接下载到本地
        logger.info("[VLM] 下载模型: {} → {}", self.model_id, self.local_dir)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self.model_id, local_dir=self.local_dir)
        return self.local_dir

    def _find_hf_cached_snapshot(self, model_id: str) -> Optional[str]:
        """在 HF cache 中查找模型快照。"""
        hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
        snapshots_dir = hf_home / f"models--{model_id.replace('/', '--')}" / "snapshots"
        if not snapshots_dir.is_dir():
            return None
        for snap in sorted(snapshots_dir.iterdir(), reverse=True):
            if snap.is_dir() and (snap / "config.json").is_file():
                return str(snap)
        return None

    def _load_model(self):
        """加载模型和 processor（延迟加载）。"""
        if self._loaded:
            return

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = self._ensure_model()

        logger.info("[VLM] 加载模型: {}", model_path)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=self.device,
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=28 * 28,
            max_pixels=self.max_pixels,
        )
        self._loaded = True
        logger.info("[VLM] 模型加载完成")

    async def describe(self, images: list[ImageInfo]) -> list[ImageInfo]:
        if not images:
            return images

        # 延迟加载模型
        self._load_model()

        for img in images:
            try:
                # 转为 PIL Image
                pil_img = PILImage.open(io.BytesIO(img.image_bytes)).convert("RGB")

                # 压缩大图
                pil_img = self._compress_image(pil_img)

                # 构建 prompt
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": IMAGE_DESCRIBE_PROMPT},
                            {"type": "image"},
                        ],
                    }
                ]

                # 处理
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=[text],
                    images=[pil_img],
                    padding=True,
                    return_tensors="pt",
                ).to(self._model.device)

                # 生成
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=600,
                )
                result = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]

                # 提取 assistant 回复
                if "assistant" in result:
                    result = result.split("assistant")[-1].strip()

                img.description = result.strip() or "[图片]"
                logger.debug(
                    "Local VLM described image: {} -> {} chars",
                    img.mime_type,
                    len(img.description),
                )

            except Exception as e:
                logger.warning("Local VLM description failed: {}", e)
                img.description = "[图片描述生成失败]"

        return images

    def _compress_image(self, image: "PILImage.Image", max_size: int = 384) -> "PILImage.Image":
        """压缩图片到最大 384px（再高会让 VLM 解析明显变慢，再低看不清图表里的小字）。"""
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            return image.resize(new_size, PILImage.LANCZOS)
        return image

    def close(self) -> None:
        """释放模型资源。"""
        if self._loaded:
            logger.info("[VLM] 释放模型内存")
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False

            # 清理 GPU 内存
            import gc

            gc.collect()
            try:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


# ===== 工厂 =====


def create_image_describer(
    api_key: str = "",
    base_url: str = "",
    model: str = "gpt-4o-mini",
    local_model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    local_dir: Optional[str] = None,
    device: str = "auto",
) -> ImageDescriber:
    """创建图片描述器。

    自动检测：
    - 有 API Key → 使用 OpenAI API
    - 无 API Key → 使用本地 VLM
    """
    if api_key:
        return OpenAIImageDescriber(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            model=model,
        )
    else:
        return LocalVLMDescriber(
            model_id=local_model_id,
            local_dir=local_dir,
            device=device,
        )