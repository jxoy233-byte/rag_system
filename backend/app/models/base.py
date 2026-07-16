from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
     """统一基类：主键 + 创建/更新时间。"""

     id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
     created_at: Mapped[datetime] = mapped_column(
         DateTime(timezone=True), server_default=func.now(), nullable=False
     )
     updated_at: Mapped[datetime] = mapped_column(
         DateTime(timezone=True),
         server_default=func.now(),
         onupdate=func.now(),
         nullable=False,
     )

     def __repr__(self) -> str:  # pragma: no cover - debug helper
         return f"<{self.__class__.__name__} id={self.id}>"
