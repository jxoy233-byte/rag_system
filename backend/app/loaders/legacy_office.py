"""旧版 Office 文档加载器：.doc / .ppt / .xls。

- .doc → 调 antiword（`brew install antiword`）
- .ppt → 调 catppt（`brew install catdoc`）
- .xls → xlrd>=2.0.1（纯 Python）

注意：这些工具/库只保证提取文本，不保留复杂排版与图片元数据。
如果二进制不在 PATH 上，doc/ppt 会抛 RuntimeError 并附带安装提示。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.loaders.base import BaseLoader, LoadedDocument, LoadedPage

_CLI_TIMEOUT_SEC = 30

# 安装提示里直接给可复制的 brew 命令；用户在别处安装时也能照搬。
_INSTALL_HINT_ANTIWORD = (
    "未找到 `antiword` 命令。\n"
    "  安装方式（macOS）：brew install antiword\n"
    "  安装方式（Debian/Ubuntu）：apt-get install antiword\n"
    "或者改用 LibreOffice（功能更强，但依赖较重）。"
)
_INSTALL_HINT_CATPPT = (
    "未找到 `catppt` 命令（catdoc 包）。\n"
    "  安装方式（macOS）：brew install catdoc\n"
    "  安装方式（Debian/Ubuntu）：apt-get install catdoc\n"
    "或者改用 LibreOffice（功能更强，但依赖较重）。"
)


def _check_tool(name: str, hint: str) -> str:
    """检查 CLI 工具是否在 PATH 上，返回绝对路径；缺失则抛带 hint 的 RuntimeError。"""
    path = shutil.which(name)
    if not path:
        raise RuntimeError(hint)
    return path


def _run_cli(cmd: list[str], hint: str) -> str:
    """跑 CLI，stdout 解码为 utf-8 → 失败回退 latin-1。返回纯文本。"""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        timeout=_CLI_TIMEOUT_SEC,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"{cmd[0]} 解析失败 (exit={proc.returncode}): {stderr[:500] or 'unknown'}")
    raw = proc.stdout or b""
    # 优先 utf-8；legacy 工具默认 Latin-1，UTF-8 解不开会塞 U+FFFD
    # 用 errors="ignore" + 先 utf-8 后 latin-1 兼容 CJK
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def _split_into_pages(text: str) -> list[LoadedPage]:
    """按空行/换页符分页；空文档返回空列表。"""
    text = text.strip()
    if not text:
        return []
    # form-feed (\f) 是常见的“分页”标记；否则按双换行分块
    chunks: list[str] = []
    if "\f" in text:
        chunks = [c.strip() for c in text.split("\f") if c.strip()]
    else:
        # 把双换行当成段落边界
        for block in text.split("\n\n"):
            blk = block.strip()
            if blk:
                chunks.append(blk)
    if not chunks:
        chunks = [text]
    return [LoadedPage(text=chunk, page=i + 1) for i, chunk in enumerate(chunks)]


class DocLoader(BaseLoader):
    """Legacy Word .doc via antiword。"""

    extensions = (".doc",)

    def load(self, path: str | Path) -> LoadedDocument:
        p = Path(path)
        _check_tool("antiword", _INSTALL_HINT_ANTIWORD)
        text = _run_cli(["antiword", str(p)], _INSTALL_HINT_ANTIWORD)
        pages = _split_into_pages(text)
        full = "\n\n".join(pg.text for pg in pages)
        return LoadedDocument(
            text=full,
            pages=pages,
            metadata={"tool": "antiword", "char_count": len(full)},
            source=str(p),
            ext=p.suffix.lower(),
        )


class PptLoader(BaseLoader):
    """Legacy PowerPoint .ppt via catppt。

    catppt 不保留 slide 边界，所有文本输出到 page=1；元数据里标 slide 数估算。
    """

    extensions = (".ppt",)

    def load(self, path: str | Path) -> LoadedDocument:
        p = Path(path)
        _check_tool("catppt", _INSTALL_HINT_CATPPT)
        text = _run_cli(["catppt", str(p)], _INSTALL_HINT_CATPPT)
        # catppt 在每页之间插一个空行；按页拆
        # 但更稳妥：按 form-feed（部分版本支持）再回退到双换行
        pages = _split_into_pages(text)
        if not pages:
            full = ""
        else:
            full = "\n\n".join(pg.text for pg in pages)
        # 估算 slide 数：catppt 输出里 `\f` 出现次数 + 1；没有则用页数
        slide_estimate = text.count("\f") + 1 if "\f" in text else len(pages)
        return LoadedDocument(
            text=full,
            pages=pages,
            metadata={"tool": "catppt", "slide_count": slide_estimate, "char_count": len(full)},
            source=str(p),
            ext=p.suffix.lower(),
        )


class XlsLoader(BaseLoader):
    """Legacy Excel .xls via xlrd 2.x。"""

    extensions = (".xls",)

    def load(self, path: str | Path) -> LoadedDocument:
        p = Path(path)
        try:
            import xlrd
        except ImportError as e:
            raise RuntimeError(
                "缺少 xlrd 库（用于解析 .xls）。请在 backend/ 下执行 "
                "`pip install 'xlrd>=2.0.1,<3.0'`。"
            ) from e

        # xlrd 2.x 不再支持 .xlsx；这里仅 .xls
        book = xlrd.open_workbook(str(p))
        pages: list[LoadedPage] = []
        for sheet_idx in range(book.nsheets):
            sheet = book.sheet_by_index(sheet_idx)
            lines: list[str] = [f"# Sheet: {sheet.name}"]
            for row_idx in range(sheet.nrows):
                row = sheet.row(row_idx)
                cells: list[str] = []
                for cell in row:
                    cells.append(_format_xls_cell(cell, book))
                if any(c.strip() for c in cells):
                    lines.append(" | ".join(cells))
            if len(lines) > 1:
                pages.append(LoadedPage(text="\n".join(lines), section=sheet.name))

        full = "\n\n".join(pg.text for pg in pages)
        return LoadedDocument(
            text=full,
            pages=pages,
            metadata={"sheet_count": len(pages), "tool": "xlrd", "char_count": len(full)},
            source=str(p),
            ext=p.suffix.lower(),
        )


def _format_xls_cell(cell, book) -> str:
    """把 xlrd 单元格值转字符串；日期类型按 ISO 格式输出。"""
    ctype = cell.ctype
    value = cell.value
    # ctype: 0 empty, 1 text, 2 number, 3 date, 4 boolean, 5 error, 6 blank
    if ctype == 0 or ctype == 6:
        return ""
    if ctype == 1:
        return str(value).strip()
    if ctype == 4:
        return "TRUE" if value else "FALSE"
    if ctype == 5:
        return f"#ERR({value})"
    if ctype == 3:
        # date: 浮点序列日期
        try:
            from xlrd import xldate

            dt = xldate.xldate_as_datetime(value, book.datemode)
            return dt.isoformat(sep=" ")
        except Exception:
            return str(value)
    # ctype == 2 number
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
