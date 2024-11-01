import asyncio

import requests
from loguru import logger

from freegpt import LLMConfig, LLMException, LLMManager, LLMModelEnum
from freegpt_db import MessageStore


# Пример использования
async def main():
    message_store = MessageStore()
    await message_store.init_db()

    config = LLMConfig(
        temperature=0.7,
        max_tokens=4000,
        model_routing=LLMModelEnum.ALL_MODELS,
        timeout=20.0,
    )

    llm_manager = LLMManager(message_store, config)

    try:
        image = requests.get(
            "https://optifyhub.ru/static/blog_images/ideogram.jpeg", stream=True
        ).raw

        response = await llm_manager.get_response(
            system_message="Отвечай на русском языке.",
            message="Что тут изображено",
            dialog_id="dialog_2",
            context_messages=10,
            image=image,
        )

        logger.info(f"Ответ: {response}")
    except LLMException as e:
        logger.error(f"Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
