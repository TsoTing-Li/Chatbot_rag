from typing import Union

from tools.logger import config_logger

# init log
LOGGER = config_logger(
    log_name="short_term_mem.log",
    logger_name="short_term_mem",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class ChatHistory:
    """
    Short term memory for chat history.

    This class provides functionality to store, retrieve, and manage short term memory
    of chat conversations categorized by topics.

    Methods:
        forget(topic: str, idx: int) -> None:
            Delete the memory at the specified index for a given topic.

        remember(topics: List[str], user_prompt: str, bot_answer: str) -> None:
            Add a new conversation to the short term memory.

        get(topics: Union[List[str], None] = None) -> Dict[str, List[Dict[str, str]]]:
            Retrieve memory for given topics or all topics if none are specified.
    """

    def __init__(self, topics: list, limit_len: int = 20) -> None:
        """
        Initialize the short term memory with given topics and a length limit.

        Args:
            topics (List[str], optional): List of default topics.
            limit_len (int, optional): Maximum number of conversations to store per topic. Defaults to 20.
        """
        LOGGER.info("Init short term memory...")
        self.limit_len = limit_len

        self.chat_history = self._build(topics=topics)
        LOGGER.info(f"Success init short term memory : {self.chat_history }")

    def forget(self, topic: str, idx: int) -> None:
        """
        Delete the memory at the specified index for a given topic.

        Args:
            topic (str): The topic of the conversation.
            idx (int): The index of the chat history to delete.
        """
        forget_info = self.chat_history[topic].pop(idx)
        LOGGER.info(f"Success delete '{forget_info}' from '{topic}' ")

    def _build(self, topics: list) -> dict:
        """
        Build the memory area for given topics.

        Args:
            topics (List[str]): List of topics.

        Returns:
            Dict[str, List[Dict[str, str]]]: Initialized memory area.
        """
        chat_history = {}
        for topic in topics:
            chat_history[topic] = []
        return chat_history

    def _check(self, topic: str) -> None:
        """
        Ensure the length of chat history does not exceed the limit.

        Args:
            topic (str): The topic of the conversation.
        """
        if len(self.chat_history[topic]) > self.limit_len:
            self.forget(idx=0)

    def remember(self, topics: list, user_prompt: str, bot_answer: str) -> None:
        """
        Add a new conversation to the short term memory.

        Args:
            topics (List[str]): List of conversation topics.
            user_prompt (str): The user's question.
            bot_answer (str): The bot's generated answer.
        """

        conversation = {"user": user_prompt, "bot": bot_answer}
        for topic in topics:
            self.chat_history[topic].append(conversation)
            LOGGER.info(f"Success add '{conversation}' to '{topic}' ")
            self._check(topic=topic)

    def get(self, topics: Union[list] = None) -> dict:
        """
        Retrieve memory for given topics or all topics if none are specified.

        Args:
            topics (Union[List[str], None], optional): List of conversation topics. Defaults to None.

        Returns:
            Dict[str, List[Dict[str, str]]]: The requested memory.
        """
        if topics:
            result = {
                topic: self.chat_history[topic]
                for topic in topics
                if self.chat_history[topic]
            }

            if not result:
                LOGGER.warning("Success get empty from short term memory ")
            LOGGER.info(f"Success get '{result}' from short term memory ")
            return result

        LOGGER.info(f"Success get all '{self.chat_history}' from short term memory ")
        return self.chat_history


if __name__ == "__main__":
    topics = ["Product", "Sale", "fruit", "sport", "Other"]
    s_memory = ChatHistory(topics=topics, limit_len=20)
    print(s_memory.get())
    s_memory.remember(topics=["Other"], user_prompt="how r u ?", bot_answer="I'm fine!")

    print(s_memory.get(topics=["Other"]))
