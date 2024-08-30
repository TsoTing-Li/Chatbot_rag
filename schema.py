import re
from typing import List

from fastapi import File, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator


class PostChat(BaseModel):
    username: str
    department: str
    prompt: str | None
    friendly: str | None

    @model_validator(mode="after")
    def check(self: "PostChat") -> "PostChat":
        if bool(re.search(r"[^a-zA-Z0-9_\-\s/\u4E00-\u9FFF]+", self.username)) is True:
            raise RequestValidationError(
                {"messages": f"username: {self.username} contain invalid characters."}
            )

        if (
            bool(re.search(r"[^a-zA-Z0-9_\-\s/\u4E00-\u9FFF]+", self.department))
            is True
        ):
            raise RequestValidationError(
                {
                    "messages": f"department: {self.department} contain invalid characters."
                }
            )

        return self


class PostSubmit(BaseModel):
    username: str
    department: str

    @model_validator(mode="after")
    def check(self: "PostSubmit") -> "PostSubmit":
        if bool(re.search(r"[^a-zA-Z0-9_\-\s/\u4E00-\u9FFF]+"), self.username) is True:
            raise RequestValidationError(
                {"messages": f"username: {self.username} contain invalid characters."}
            )

        if (
            bool(re.search(r"[^a-zA-Z0-9_\-\s/\u4E00-\u9FFF]+"), self.department)
            is True
        ):
            raise RequestValidationError(
                {
                    "messages": f"department: {self.department} contain invalid characters."
                }
            )
        return self


class PostReport(BaseModel):
    username: str
    department: str
    feedback: str

    @model_validator(mode="after")
    def check(self: "PostReport") -> "PostReport":
        if bool(re.search(r"[^a-zA-Z0-9_\-\s/\u4E00-\u9FFF]+"), self.username) is True:
            raise RequestValidationError(
                {"messages": f"username: {self.username} contain invalid characters."}
            )

        if (
            bool(re.search(r"[^a-zA-Z0-9_\-\s/\u4E00-\u9FFF]+"), self.department)
            is True
        ):
            raise RequestValidationError(
                {
                    "messages": f"department: {self.department} contain invalid characters."
                }
            )

        if bool(re.search(r"[^a-zA-Z0-9_\-\s/\u4E00-\u9FFF]+"), self.feedback) is True:
            raise RequestValidationError(
                {"messages": f"feedback: {self.feedback} contain invalid characters."}
            )
        return self


class PostUpload(BaseModel):
    files: List[UploadFile] = File(...)
    username: str
    department: str

    @model_validator(mode="after")
    def check(self: "PostUpload") -> "PostUpload":
        for file in self.files:
            if file.content_type != "application/pdf":
                raise RequestValidationError(
                    {
                        "messages": f"Upload files content invalid format: {file.filename}, only support PDF file."
                    }
                )
        return self
