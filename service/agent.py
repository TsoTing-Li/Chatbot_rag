from collections.abc import Generator
from typing import Optional

from core.handler.text_to_text import GenText
from core.handler.topics_classifier import TopicsClassifier
from core.models.pattern import (
    ImageEmbedding,
    Text2Text,
    TextEmbedding,
    TopicsClassification,
)
from core.prompt.main import PromptEngineerService
from tools.logger import config_logger

from .pools.memory import MemoryService
from .pools.retriever import RetrieverService


class Agent:
    """
    Agent class for handling chat interactions and integrating various services.

    This class uses text generation, memory management, retrieval, and topic classification
    services to process chat prompts and generate responses.

    Methods:
        chat(prompt: str, file: Optional[Image.Image] = None) -> str:
            Handle chat prompt with optional image input and generate a response.
    """

    def __init__(
        self,
        gen_text_model: Text2Text,
        text_emb_model: TextEmbedding,
        topics_classifier_service: TopicsClassification,
        topics: list = None,
    ) -> None:
        """
        Initialize the Agent with various models and services.

        Args:
            gen_text_model (Text2Text): The text generation model.
            text_emb_model (TextEmbedding): The text embedding model.
            img_emb_model (ImageEmbedding): The image embedding model.
            topics_classifier_model (TopicsClassification): The topics classifier model.
            topics (List[str], optional): List of default topics. Defaults to predefined list.
        """
        if not topics:
            topics = [
                "Product Information",
                "Pricing and Promotions",
                "Purchasing and Orders",
                "After-sales Service",
                "Company Information",
            ]
        self.gentxt_service = GenText(model=gen_text_model)
        self.memory_service = MemoryService(model=gen_text_model, topics=topics)
        self.retriever_service = RetrieverService(
            text_emb_model=text_emb_model,
        )
        self.topics_classifier_service = TopicsClassifier(
            model=topics_classifier_service,
            topics=topics,
            url=topics_classifier_service.url,
        )
        self.prompt_engineer = PromptEngineerService()

    def chat(
        self,
        log: config_logger,
        prompt: str,
        friendly: str = None,
    ) -> Generator[str]:
        """
        Handle chat prompt with optional image input and generate a response.

        Args:
            log (config_logger): logger.
            prompt (str): The chat prompt from the user.
            friendly (str): Friendly say hello at first time.
        """

        try:
            log.info("Start chat!")
            log.info(f"User prompt: '{prompt}'.")
            topics = self.topics_classifier_service.run(sentence=prompt)
            log.info(f"Topics: '{topics}'.")
            conversation_history = self.memory_service.get_chat_history(topics=topics)
            log.info(f"Conversation history: '{conversation_history}'.")
            instruction = self.memory_service.get_instruction()
            log.info(f"Instruction: '{instruction}'.")
            retriever = self.retriever_service.search(data=prompt)
            log.info(f"Retriever: '{retriever}'.")
            user_prompt = self.prompt_engineer.generate(
                history=conversation_history,
                retrieval=retriever,
                prompt=prompt,
                instruction=instruction,
            )
            system_prompt = self.prompt_engineer.instruction_content()
            final_prompt = [
                {"role": "system", "content": system_prompt[0]},
                {"role": "system", "content": system_prompt[1]},
                {"role": "system", "content": system_prompt[2]},
                {"role": "user", "content": user_prompt},
            ]
            if friendly:
                final_prompt.insert(0, {"role": "system", "content": friendly})
            log.info(f"Final prompt: '{str(final_prompt)}'.")

        except BaseException:
            log.error("Can not preprocess prompt")
            raise RuntimeError
        
        try:
            content = ""
            for data in self.gentxt_service.run(data=final_prompt):
                content += data
                yield data
        except BaseException:
            log.error("Can not execute gentxt service")
            raise RuntimeError
        
        log.info(f"Response: '{content}'.")

        try:
            self.memory_service.remember(
                topics=topics, user_prompt=prompt, bot_answer=content
            )
        except BaseException:
            log.error("Can not execute memory service")
            raise RuntimeError