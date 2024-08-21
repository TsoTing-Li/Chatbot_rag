from typing import List, Union

from jinja2 import Template

from tools.logger import config_logger

# init log
LOGGER = config_logger(
    log_name="prompt.log",
    logger_name="prompt",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class PromptEngineerService:
    """
    Prompt engineering service.

    This class provides functionality to generate prompts for various purposes such as
    conversation history summarization and generating answers to user questions.

    Methods:
        summary_history(chat_history: Dict[str, List[Dict[str, str]]]) -> List[ChatMessage]:
            Generate a summary of the conversation history.

        generate(
            history: Union[str, bool],
            retrieval: Union[str, bool],
            prompt: str,
            instruction: Union[List[str], None] = None
        ) -> List[ChatMessage]:
            Generate a prompt based on conversation history, retrieval information, and user question.
    """

    def __init__(self) -> None:
        """
        Initialize the Service with a chat prompt builder and predefined templates.
        """
        # self.builder = PromptTemplate()

        self.template = {
            "chat": """
{% if instruction %}
Instructions:
{% for inst in instruction %}
- {{ inst }}
{% endfor %}
{% endif %}

{% if conversation_history %}
Conversation History:
{{ conversation_history }}
{% endif %}

{% if retriever_info %}
Retriever's Information:
{{ retriever_info }}
{% endif %}

Question: {{ question }}
Answer:
""",
            "summary": """
Please provide a concise summary of the following conversation history. Focus on the key points and important details mentioned.

{% for topic, conversations in history.items() %}
Topic: {{ topic }}
Conversation History:
{% for conversation in conversations %}
User ask: {{ conversation.user }} Bot answer: {{ conversation.bot }}
{% endfor %}
{% endfor %}

Overall Summary:
""",
        }
        LOGGER.info("Success init prompt !")

    def summary_history(self, chat_history: dict) -> str:
        """
        Generate a summary of the conversation history.

        Args:
            chat_history (Dict[str, List[Dict[str, str]]]): The conversation history to summarize.

        Returns:
            str: The prompt for summarizing the conversation history.
        """

        template = Template(self.template["summary"])
        prompt = template.render(history=chat_history)
        LOGGER.info(f"Get history summary prompt : {prompt}")
        return prompt

    def instruction_content(self) -> List[str]:
        return [
            "You are a chatbot which name iVIT-Chatbot",
            "If user didn't ask for for more detail please answer question within 100 words",
            "If you don't know just tell user I don't know",
        ]

    def generate(
        self,
        history: Union[str, bool],
        retrieval: Union[str, bool],
        prompt: str,
        instruction: Union[list, None] = None,
    ) -> str:
        """
        Generate a prompt based on conversation history, retrieval information, and user question.

        Args:
            history (Union[str, bool]): Conversation history.
            retrieval (Union[str, bool]): Information retrieved from a database.
            prompt (str): User question.
            instruction (Union[List[str], None], optional): System instructions. Defaults to None.

        Returns:
            str: The generated prompt for answering the user question.
        """
        template = Template(self.template["chat"])
        prompt = template.render(
            instruction=instruction,
            conversation_history=history,
            retriever_info=retrieval,
            question=prompt,
        )

        LOGGER.info(f"Get generate answer prompt : {prompt}")
        return prompt


if __name__ == "__main__":
    prompt_engineering = PromptEngineerService()
    # summary_prompt = prompt_engineering.summary_history()
    # print(summary_prompt)
    prompt = prompt_engineering.generate(
        history="user introduces himself as jay and asks if the assistant can help him. The assistant greets jay and asks how it can assist him. Jay reveals that his job is to sell 3TE7, and the assistant supports this decision. Jay then inquires about who he can contact to purchase 3TE7, and the assistant suggests reaching out to jay.",
        retrieval="yes 123",
        prompt="ggg",
    )

    print(prompt)
