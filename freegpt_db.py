from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import aiosqlite
from loguru import logger


class MessageRole(Enum):
    """Доступные роли в диалоге"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    @classmethod
    def values(cls) -> List[str]:
        return [role.value for role in cls]


@dataclass
class Message:
    """Структура сообщения"""

    role: str
    content: str
    dialog_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "dialog_id": self.dialog_id,
            "timestamp": self.timestamp,
        }


class MessageStoreException(Exception):
    """Базовое исключение для MessageStore"""

    pass


class MessageStore:
    """Хранилище сообщений с асинхронным доступом"""

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dialog_id TEXT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """

    CREATE_INDEXES_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_dialog_timestamp ON messages(dialog_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)",
    ]

    def __init__(self, db_path: str = "messages.db"):
        self.db_path = db_path

    async def _validate_role(self, role: str) -> None:
        """Проверка корректности роли"""
        if role not in MessageRole.values():
            raise MessageStoreException(
                f"Invalid role: {role}. Must be one of {MessageRole.values()}"
            )

    async def init_db(self) -> None:
        """Инициализация базы данных и создание индексов"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(self.CREATE_TABLE_SQL)
                for index_sql in self.CREATE_INDEXES_SQL:
                    await db.execute(index_sql)
                await db.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise MessageStoreException(f"Database initialization failed: {str(e)}")

    async def save_message(
        self, role: str, content: str, dialog_id: Optional[str] = None
    ) -> None:
        """Сохранение сообщения в базу"""
        await self._validate_role(role)

        if not content.strip():
            raise MessageStoreException("Content cannot be empty")

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO messages (dialog_id, role, content) VALUES (?, ?, ?)",
                    (dialog_id, role, content),
                )
                await db.commit()
                logger.debug(f"Message saved: role={role}, dialog_id={dialog_id}")
        except Exception as e:
            logger.error(f"Failed to save message: {str(e)}")
            raise MessageStoreException(f"Failed to save message: {str(e)}")

    async def get_messages(
        self,
        dialog_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order: str = "ASC",
    ) -> List[Dict[str, Any]]:
        """Получение сообщений с пагинацией"""
        if order not in ["ASC", "DESC"]:
            raise MessageStoreException("Order must be either 'ASC' or 'DESC'")

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                query = ["SELECT role, content FROM messages"]
                params = []

                if dialog_id:
                    query.append("WHERE dialog_id = ?")
                    params.append(dialog_id)

                query.append(f"ORDER BY timestamp {order}")

                if limit is not None:
                    query.append("LIMIT ?")
                    params.append(limit)

                if offset is not None:
                    query.append("OFFSET ?")
                    params.append(offset)

                async with db.execute(" ".join(query), params) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get messages: {str(e)}")
            raise MessageStoreException(f"Failed to get messages: {str(e)}")

    async def get_last_messages(
        self, limit: Optional[int] = None, dialog_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Получение последних сообщений в хронологическом порядке"""
        messages = await self.get_messages(
            dialog_id=dialog_id, limit=limit, order="DESC"
        )
        return list(reversed(messages))

    async def get_all_messages_iter(self) -> AsyncIterator[Dict[str, Any]]:
        """Получение всех сообщений построчно через асинхронный итератор"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM messages ORDER BY timestamp ASC"
                ) as cursor:
                    async for row in cursor:
                        yield dict(row)
        except Exception as e:
            logger.error(f"Failed to iterate messages: {str(e)}")
            raise MessageStoreException(f"Failed to iterate messages: {str(e)}")

    async def delete_dialog(self, dialog_id: str) -> None:
        """Удаление диалога"""
        if not dialog_id:
            raise MessageStoreException("dialog_id cannot be empty")

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "DELETE FROM messages WHERE dialog_id = ?", (dialog_id,)
                )
                await db.commit()
                logger.info(f"Dialog deleted: {dialog_id}")
        except Exception as e:
            logger.error(f"Failed to delete dialog: {str(e)}")
            raise MessageStoreException(f"Failed to delete dialog: {str(e)}")

    async def count_messages(self, dialog_id: Optional[str] = None) -> int:
        """Подсчет количества сообщений"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = "SELECT COUNT(*) FROM messages"
                params = []

                if dialog_id:
                    query += " WHERE dialog_id = ?"
                    params.append(dialog_id)

                async with db.execute(query, params) as cursor:
                    (count,) = await cursor.fetchone()
                    return count
        except Exception as e:
            logger.error(f"Failed to count messages: {str(e)}")
            raise MessageStoreException(f"Failed to count messages: {str(e)}")
