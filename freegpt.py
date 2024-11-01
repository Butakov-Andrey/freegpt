import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from g4f.client import Client
from loguru import logger

from freegpt_db import MessageStore


class LLMVisionModelEnum(Enum):
    """Поддерживаемые модели"""

    BLACKBOXAI = "blackboxai"


class LLMModelEnum(Enum):
    """Поддерживаемые модели"""

    CLAUDE_3_5_SONNET = "claude-3.5-sonnet"
    CLAUDE_3_OPUS = "claude-3-opus"
    GEMINI_PRO = "gemini-pro"
    O1_MINI = "o1-mini"
    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    GEMINI_FLASH = "gemini-flash"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    DEEPSEEK = "deepseek"
    GPT_4O_MINI = "gpt-4o-mini"
    MISTRAL_LARGE = "mistral-large"
    QWEN_2_72B = "qwen-2-72b"
    BLACKBOXAI_PRO = "blackboxai-pro"
    LLAMA_3_1_405B = "llama-3.1-405b"
    LLAMA_3_1_70B = "llama-3.1-70b"
    LLAMA_3_2_90B = "llama-3.2-90b"
    YI_34B = "yi-34b"
    MIXTRAL_8X22B = "mixtral-8x22b"
    GEMMA_2B_27B = "gemma-2b-27b"
    GPT_4 = "gpt-4"

    @classmethod
    def all_models(cls) -> List["LLMModelEnum"]:
        """Получить список всех доступных моделей"""
        return list(cls)

    # Создаем специальный атрибут для обозначения всех моделей
    ALL_MODELS = "all_models"


ModelRoutingType = Union[List[LLMModelEnum], str]


@dataclass
class LLMConfig:
    """Конфигурация для LLM запросов"""

    temperature: float = 0.7
    max_tokens: int = 4000
    model_routing: ModelRoutingType = None
    timeout: float = 30.0

    def __post_init__(self):
        if self.model_routing is None or self.model_routing == LLMModelEnum.ALL_MODELS:
            self.model_routing = LLMModelEnum.all_models()
        elif isinstance(self.model_routing, str):
            raise ValueError(
                f"Неверное значение model_routing: {self.model_routing}. "
                f"Используйте LLMModelEnum.ALL_MODELS или список моделей"
            )

    def get_models(self) -> List[LLMModelEnum]:
        """Получить список моделей для использования"""
        return self.model_routing


class LLMException(Exception):
    """Базовое исключение для LLM ошибок"""

    pass


class ModelFailedException(LLMException):
    """Исключение при ошибке модели"""

    pass


class ModelTimeoutException(ModelFailedException):
    """Исключение при превышении времени ожидания ответа от модели"""

    pass


class AllModelsFailedException(LLMException):
    """Исключение, когда все модели завершились с ошибкой"""

    pass


@dataclass
class Message:
    """Структура сообщения"""

    role: str
    content: str
    has_image: bool = False


class LLMManager:
    """Менеджер для работы с LLM моделями"""

    def __init__(self, message_store: MessageStore, config: Optional[LLMConfig] = None):
        self.message_store = message_store
        self.config = config or LLMConfig()
        self.client = Client()

    async def try_model(
        self,
        model: LLMModelEnum,
        messages: List[Dict[str, str]],
        image: Optional[Any] = None,
    ) -> Optional[str]:
        """Попытка получить ответ от конкретной модели"""
        try:
            kwargs = {
                "model": model.value,
                "temperature": self.config.temperature,
                "n": 1,
                "max_tokens": self.config.max_tokens,
                "messages": messages,
            }
            if image:
                kwargs["image"] = image

            response = await asyncio.wait_for(
                self.client.chat.completions.async_create(**kwargs),
                timeout=self.config.timeout,
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            logger.error(
                f"Таймаут для модели {model.value} после {self.config.timeout} секунд"
            )
            raise ModelTimeoutException(
                f"Модель {model.value} не ответила за {self.config.timeout} секунд"
            )
        except Exception as e:
            logger.error(f"Ошибка при использовании модели {model.value}: {str(e)}")
            raise ModelFailedException(
                f"Модель {model.value} завершилась с ошибкой: {str(e)}"
            )

    def _prepare_messages(
        self, system_message: str, context: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Подготовка сообщений для API"""
        messages = [{"role": "system", "content": system_message}]
        messages.extend(
            [{"role": msg["role"], "content": msg["content"]} for msg in context]
        )
        return messages

    async def get_response(
        self,
        system_message: str,
        message: str,
        dialog_id: Optional[str] = None,
        context_messages: int = 5,
        image: Optional[Any] = None,
    ) -> str:
        """Получение ответа от LLM с автоматическим роутингом"""
        # Сохраняем сообщение пользователя
        await self.message_store.save_message(
            "user", message if message else "[Изображение]", dialog_id
        )

        # Получаем контекст
        context = await self.message_store.get_last_messages(
            limit=context_messages,
            dialog_id=dialog_id,
        )

        # Формируем сообщения
        messages = self._prepare_messages(system_message, context)

        # Если есть изображение, используем только blackboxai
        if image:
            try:
                response = await self.try_model(
                    LLMVisionModelEnum.BLACKBOXAI, messages, image
                )
                if response:
                    logger.info("Успешный ответ от модели blackboxai")
                    await self.message_store.save_message(
                        "assistant", response, dialog_id
                    )
                    return response
            except (ModelFailedException, ModelTimeoutException) as e:
                logger.error(str(e))
                raise AllModelsFailedException(
                    "Модель для обработки изображений недоступна"
                )

        # Если нет изображения, используем стандартный роутинг
        errors = []
        for model in self.config.model_routing:
            logger.info(f"Пробуем модель: {model.value}")
            try:
                response = await self.try_model(model, messages)
                if response:
                    logger.info(f"Успешный ответ от модели: {model.value}")
                    await self.message_store.save_message(
                        "assistant", response, dialog_id
                    )
                    return response
            except (ModelFailedException, ModelTimeoutException) as e:
                logger.warning(str(e))
                errors.append(str(e))
                continue

        error_msg = "Все модели завершились с ошибкой:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise AllModelsFailedException(error_msg)
        raise AllModelsFailedException(error_msg)
