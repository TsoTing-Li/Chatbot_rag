import json
from pathlib import Path
from typing import List

from datasets import Dataset
from pydantic import BaseModel, model_validator

from tools.logger import config_logger

from .error_msg import (
    ImageFolderNotFoundError,
    ImageNotFoundError,
    MissingKeysError,
    MultipleJsonFilesError,
    NoJsonFilesError,
)

# init log
LOGGER = config_logger(
    log_name="faiss_data.log",
    logger_name="faiss_data",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class DataFormat(BaseModel):
    """
    DataFormat class for validating and storing image and description data.

    Attributes:
        images (List[str]): List of image file paths.
        describe (List[str]): List of descriptions corresponding to the images.

    Methods:
        check_equal_length() -> None:
            Validate that the lengths of the images and describe lists are equal.
    """

    images: List[str]
    describe: List[str]

    @model_validator(mode="after")
    def check_equal_length(self) -> None:
        """
        Validate that the lengths of the images and describe lists are equal.

        Raises:
            ValueError: If the lengths of the images and describe lists are not equal.
        """
        if len(self.images) != len(self.describe):
            LOGGER.error("Length of images and describe lists must be the same")
            raise ValueError("Length of images and describe lists must be the same")


class Process:
    """
    Process class for handling and converting raw data into a Dataset.

    Attributes:
        dataset (Dataset): The dataset created from the raw data.

    Methods:
        save(file_path: str) -> None:
            Save the dataset to disk.
    """

    def __init__(self) -> None:
        """
        Initialize the Process class with raw data.

        Args:
            raw_data (dict): The raw data containing image paths and descriptions.
        """

    def _check_path(self, data_folder: str) -> bool:
        """
        Check if the data_folder contains only one JSON file, has a folder for images,
        and the images specified in the JSON file exist in the image folder.

        Args:
            data_folder (str): The path to the data_folder to be checked.

        Returns:
            bool: True if the data_folder meets all criteria, False otherwise.

        Raises:
            MultipleJsonFilesError: If there are multiple JSON files in the data_folder.
            NoJsonFilesError: If there are no JSON files in the data_folder.
            MissingKeysError: If the JSON file is missing required keys.
            ImageFolderNotFoundError: If no folder containing images is found.
            ImageNotFoundError: If images specified in the JSON file do not exist.
        """
        directory_path = Path(data_folder)

        json_files = list(directory_path.glob("*.json"))

        if len(json_files) > 1:
            LOGGER.error("Multiple JSON files found in the data_folder.")
            raise MultipleJsonFilesError(
                "Multiple JSON files found in the data_folder."
            )
        elif len(json_files) == 0:
            LOGGER.error("No JSON files found in the data_folder.")
            raise NoJsonFilesError("No JSON files found in the data_folder.")

        with open(json_files[0]) as json_file:
            data = json.load(json_file)

        if "images" not in data or "describe" not in data:
            LOGGER.error("JSON file is missing required keys 'images' 或 'describe'.")
            raise MissingKeysError(
                "JSON file is missing required keys 'images' 或 'describe'."
            )

        image_folder_found = False
        image_folder = None
        all_images_exist = False

        for item in directory_path.iterdir():
            if item.is_dir():
                image_files = (
                    list(item.glob("*.png"))
                    + list(item.glob("*.jpg"))
                    + list(item.glob("*.jpeg"))
                )
                if image_files:
                    all_images_exist = all(
                        (item / Path(image_path)).exists()
                        for image_path in data["images"]
                    )
                    if all_images_exist:
                        image_folder_found = True
                        image_folder = item
                        break
        for image_path in data["images"]:
            print(item / Path(image_path))
            print(item)
        if not image_folder_found:
            LOGGER.error("No folder containing images was found.")
            raise ImageFolderNotFoundError("No folder containing images was found.")
        if not all_images_exist:
            LOGGER.error(
                "Some images specified in the JSON file do not exist in the image folder."
            )
            raise ImageNotFoundError(
                "Some images specified in the JSON file do not exist in the image folder."
            )

        absolute_image_paths = [
            str((image_folder / Path(image_path).name).resolve())
            for image_path in data["images"]
        ]
        data["images"] = absolute_image_paths

        with open(json_files[0], "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

        return True

    def _turn2datasets(self, data: dict) -> Dataset:
        """
        Convert the DataFormat object into a Dataset.

        Args:
            data (DataFormat): The validated data.

        Returns:
            Dataset: The resulting Dataset object.
        """
        LOGGER.info("Success transfer data type dict to dataset")
        return Dataset.from_dict(data.dict())

    def run(
        self, data_folder: str, file_path: str = "./database/faiss/embeddings/data"
    ) -> Dataset:
        """
        Turn to the dataset format to Dataset and save to disk.

        Args:
            data_folder (str) : The path where the dataset will be load.
            file_path (str): The path where the dataset will be saved.
        Returns:
            Dataset: Dataset object.
        """

        self._check_path(data_folder=data_folder)
        LOGGER.info("Success check data folder!")

        sources = list(Path(data_folder).glob("**/*.json"))

        # Only one json file in folder
        with open(sources[0]) as json_file:
            raw_data = json.load(json_file)
        dataset = self._turn2datasets(data=DataFormat(**raw_data))
        LOGGER.info(f"Success turn to the dataset format to Dataset : {raw_data}")

        dataset.save_to_disk(file_path)
        LOGGER.info(f"Success save data to {file_path}")
        return dataset
