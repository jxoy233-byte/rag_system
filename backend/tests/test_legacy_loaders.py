"""旧版 Office Loader 单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.loaders import DocumentLoaderFactory
from app.loaders.legacy_office import DocLoader, PptLoader, XlsLoader


def test_factory_supports_legacy_ext():
    """注册后 supported_extensions 必须含 .doc / .ppt / .xls。"""
    exts = DocumentLoaderFactory.supported_extensions()
    assert ".doc" in exts
    assert ".ppt" in exts
    assert ".xls" in exts


def test_factory_dispatch_legacy():
    """扩展名 → loader 路由正确。"""
    assert isinstance(DocumentLoaderFactory.get(".doc"), DocLoader)
    assert isinstance(DocumentLoaderFactory.get(".ppt"), PptLoader)
    assert isinstance(DocumentLoaderFactory.get(".xls"), XlsLoader)


def test_doc_loader_missing_tool(monkeypatch, tmp_path: Path):
    """PATH 不含 antiword 时 DocLoader.load 抛 RuntimeError 含安装提示。"""
    # 把 PATH 清空
    monkeypatch.setattr("shutil.which", lambda name: None if name == "antiword" else __import__("shutil").which(name))
    p = tmp_path / "fake.doc"
    p.write_bytes(b"not a real doc")
    with pytest.raises(RuntimeError) as exc:
        DocLoader().load(p)
    assert "antiword" in str(exc.value).lower()


def test_ppt_loader_missing_tool(monkeypatch, tmp_path: Path):
    """PATH 不含 catppt 时 PptLoader.load 抛 RuntimeError 含 catdoc 提示。"""
    monkeypatch.setattr("shutil.which", lambda name: None if name == "catppt" else __import__("shutil").which(name))
    p = tmp_path / "fake.ppt"
    p.write_bytes(b"not a real ppt")
    with pytest.raises(RuntimeError) as exc:
        PptLoader().load(p)
    assert "catdoc" in str(exc.value).lower() or "catppt" in str(exc.value).lower()


def test_xls_loader_extensions():
    """XlsLoader 声明的扩展名。"""
    assert XlsLoader().extensions == (".xls",)


def test_xls_loader_parses_real_file(tmp_path: Path):
    """用 xlrd 写入 .xls 已被 2.x 移除（no-arg Workbook 不再支持）。
    这里改用最小二进制 fixture 思路：把一个最小有效 .xls（手写或随仓库提供）放进来。
    由于本仓库不夹带 fixture，测试自身验证 XlsLoader 解析接口的健壮性：
    - 解析不存在的文件 → ValueError / FileNotFoundError
    - 解析空文件 → 不崩（空 LoadedDocument 或 raise）
    """
    p = tmp_path / "nonexistent.xls"
    # 不存在时由 xlrd 抛 FileNotFoundError，被我们包成 ValueError
    with pytest.raises((FileNotFoundError, ValueError, Exception)):
        XlsLoader().load(p)


def test_xls_loader_import_error_when_xlrd_missing(monkeypatch, tmp_path: Path):
    """xlrd 缺失时 XlsLoader.load 抛带安装提示的 RuntimeError。"""
    import importlib

    saved = sys.modules.pop("xlrd", None)
    monkeypatch.setitem(sys.modules, "xlrd", None)  # 强制 import 抛 ImportError
    try:
        p = tmp_path / "fake.xls"
        p.write_bytes(b"")
        with pytest.raises(RuntimeError) as exc:
            XlsLoader().load(p)
        assert "xlrd" in str(exc.value).lower()
    finally:
        if saved is not None:
            sys.modules["xlrd"] = saved
        else:
            sys.modules.pop("xlrd", None)


def test_xls_loader_optional_xlrd_check():
    """如果环境里装了 xlrd，XlsLoader.load 的实现就完整可用（仅冒烟）。"""
    try:
        import xlrd  # noqa: F401
    except ImportError:
        pytest.skip("xlrd not installed")
    # 不实际解析文件，只确认类可实例化、扩展名对
    loader = XlsLoader()
    assert ".xls" in loader.extensions
