import copy
import json
import logging
import os
import time

from tools.logger import config_logger


class UserHandler:
    """
    UserHandler class.

    This class handles user information and logging for a user feedback system.

    Methods:
        register(username: str, department: str) -> dict:
            Register a new user.

        check(username: str, department: str) -> bool:
            Check if a user is already registered.

        get(username: str, department: str) -> dict:
            Get the logger for a user.
    """

    def __init__(self, user_info_path: str = "./feedback/user_info.json") -> None:
        """
        Initialize the UserHandler class.

        Args:
            user_info_path (str): Path to the user information JSON file. Defaults to "./feedback/user_info.json".
        """
        self.users_info = self._load(path=user_info_path)

    def _load(self, path) -> dict:
        """
        Load user information from a JSON file.

        Args:
            path (str): Path to the user information JSON file.

        Returns:
            dict: A dictionary containing user information.
        """
        users_info = {}
        if os.path.exists(path):
            with open(path) as json_file:
                users_info = json.load(json_file)

            users_info_copy = copy.deepcopy(users_info)
            for department, user_info in users_info_copy.items():
                users_info[department]["log"] = self._create_log(
                    username=user_info["name"], department=department
                )
                now_time = time.time()
                if user_info.__contains__("create_time"):
                    users_info[department]["reload_time"] = now_time

                else:
                    users_info[department]["create_time"] = now_time

        return users_info

    def _create_log(self, username: str, department: str) -> logging.Logger:
        """
        Create a logger for a user.

        Args:
            username (str): The username.
            department (str): The department the user belongs to.

        Returns:
            logging.Logger: Configured logger for the user.
        """
        log = config_logger(
            log_name=f"{department}_{username}.log",
            logger_name=f"{department}_{username}",
            default_folder="./feedback",
            write_mode="w",
            level="debug",
        )
        return log

    def _save(
        self,
        users_info: dict,
        save_folder: str = "./feedback/",
        file_name: str = "user_info.json",
    ) -> None:
        """
        Save user information to a JSON file.

        Args:
            users_info (dict): A dictionary containing user information.
            save_folder (str): Folder to save the JSON file. Defaults to "./feedback/".
            file_name (str): Name of the JSON file. Defaults to "user_info.json".
        """
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        file_path = os.path.join(save_folder, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(users_info, f, ensure_ascii=False, indent=4)

    def register(self, username: str, department: str) -> dict:
        """
        Register a new user.

        Args:
            username (str): The username to register.
            department (str): The department the user belongs to.

        Returns:
            dict: Updated user information dictionary.
        """
        username = username.lower()
        department = department.lower()
        now_time = time.time()
        log = self._create_log(username=username, department=department)
        self.users_info.update(
            {department: {"name": username, "create_time": now_time}}
        )
        if self.users_info[department].__contains__("log"):
            del self.users_info[department]["log"]
        self._save(users_info=self.users_info)
        self.users_info[department]["log"] = log

    def check(self, username: str, department: str) -> bool:
        """
        Check if a user is already registered.

        Args:
            username (str): The username to check.
            department (str): The department the user belongs to.

        Returns:
            bool: True if the user is registered, False otherwise.
        """
        username = username.lower()
        department = department.lower()
        if department in self.users_info:
            if self.users_info[department]["name"] == username:
                return True
        return False

    def get(self, username: str, department: str) -> dict:
        """
        Get the logger for a user.

        Args:
            username (str): The username.
            department (str): The department the user belongs to.

        Returns:
            dict: The logger for the user.
        """
        username = username.lower()
        department = department.lower()

        return self.users_info[department]["log"]
