from typing import Union

from tools.logger import config_logger

# init log
LOGGER = config_logger(
    log_name="long_term_mem.log",
    logger_name="long_term_mem",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class Instruction:
    """
    Instruction class for managing long-term memory commands.

    This class provides methods to add, delete, and retrieve commands stored in long-term memory.

    Methods:
        add(command: str) -> None:
            Add a new command to long-term memory.

        delete(idx: Union[int, None] = None, command: Union[str, None] = None) -> None:
            Delete a command from long-term memory.

        get() -> list:
            Retrieve all commands from long-term memory.
    """

    def __init__(self) -> None:
        """
        Initialize the Instruction class with an empty command list.
        """
        LOGGER.info("Init long term memory...")
        self.instruction = []
        LOGGER.info(f"Success init long term memory : {self.instruction}")

    def add(self, command: str) -> None:
        """
        Add a new command to long-term memory.

        Args:
            command (str): The command to add.
        """
        self.instruction.append(command)
        LOGGER.info(f"Success add '{command}' to long term memory ")

    def delete(
        self, idx: Union[int, None] = None, command: Union[str, None] = None
    ) -> None:
        """
        Delete a command from long-term memory.

        Args:
            idx (Union[int, None], optional): The index of the command to delete. Defaults to None.
            command (Union[str, None], optional): The command to delete. Defaults to None.

        Raises:
            ValueError: If both idx and command are None.
            ValueError: If both idx and command are provided.
            Exception: If an error occurs during deletion.
        """
        if (idx is None) and (command is None):
            raise ValueError("All parameter is None!")
        elif (idx is not None) and (command is not None):
            raise ValueError("Choose one parameter to delete!")
        elif (idx is not None) and (command is None):
            self.instruction.pop(idx)
        elif (idx is None) and (command is not None):
            self.instruction.remove(command)
        else:
            raise Exception("Some error happen!")
        LOGGER.info(f"Success delete '{command}' from long term memory ")

    def get(self) -> list:
        """
        Retrieve all commands from long-term memory.

        Returns:
            list: List of commands in long-term memory.
        """
        LOGGER.info(f"Success get '{self.instruction}' from long term memory ")
        return self.instruction
