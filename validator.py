from pathlib import Path

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator


class PostUpdate(BaseModel):
    folder: str
    recreate: bool = False

    @model_validator(mode="after")
    def check(self: "PostUpdate") -> "PostUpdate":
        if not Path(self.folder).is_dir():
            raise RequestValidationError(
                {"messages": f"Folder: {self.folder} is not a valid path"}
            )

        return self
