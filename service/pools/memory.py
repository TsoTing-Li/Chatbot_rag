from typing import Union

from core.handler.text_to_text import GenText
from core.memory.long_term import Instruction
from core.memory.short_term import ChatHistory
from core.models.pattern import Text2Text
from core.prompt.main import PromptEngineerService


class MemoryService:
    """
    Service for managing conversation history and integrating with a Text2Text model.

    This service includes short-term memory for recent conversation history and
    long-term memory instructions for summarizing and retrieving important information.

    Attributes:
        gen_text_service (GenText): The text generation service used for processing prompts.
        short_term_mem (ChatHistory): Object for managing short-term conversation history.
        long_term (Instruction): Object for managing long-term memory instructions.
        prompt (PromptEngineerService): Object for creating prompts for the model.

    Methods:
        remember(topics: List[str], user_prompt: str, bot_answer: str) -> None:
            Store the conversation history in short-term memory.

        get_instruction() -> List[str]:
            Retrieve the long-term memory instructions.

        get_chat_history(topics: List[str]) -> Union[str, None]:
            Retrieve and summarize the conversation history for given topics.
    """

    def __init__(self, model: Text2Text, topics: list) -> None:
        """
        Initialize the MemoryService with a Text2Text model and other components.

        Args:
            model (Text2Text): The text generation model used for processing prompts.
            topics (List[str]): List of default topics.
        """
        self.gen_text_service = GenText(model=model)
        self.short_term_mem = ChatHistory(topics=topics)
        self.long_term = Instruction()
        self.prompt = PromptEngineerService()

    def remember(self, topics: list, user_prompt: str, bot_answer: str) -> None:
        """
        Store the conversation history in short-term memory.

        Args:
            topics (List[str]): List of topics related to the conversation.
            user_prompt (str): The user's input or question.
            bot_answer (str): The bot's response.
        """
        self.short_term_mem.remember(
            topics=topics, user_prompt=user_prompt, bot_answer=bot_answer
        )

    def get_instruction(self) -> list:
        """
        Retrieve the long-term memory instructions.

        Returns:
            List[str]: The long-term memory instructions.
        """
        return self.long_term.get()

    def get_chat_history(self, topics: list) -> Union[str, None]:
        """
        Retrieve and summarize the conversation history for given topics.

        Args:
            topics (List[str]): List of topics to filter the conversation history.

        Returns:
            Union[str, None]: A summary of the conversation history if available, otherwise None.
        """
        conversation_history = self.short_term_mem.get(topics=topics)

        if conversation_history:
            summary_his_prompt = self.prompt.summary_history(
                chat_history=conversation_history
            )

            summary_history = ""
            for data in self.gen_text_service.run(
                data=[{"role": "user", "content": summary_his_prompt}]
            ):
                summary_history += data
            return summary_history

        return None
