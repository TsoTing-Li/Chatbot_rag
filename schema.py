from pydantic import BaseModel
from fastapi import UploadFile

class PostChat(BaseModel):
    username: str
    department: str
    file: UploadFile | None
    prompt: str | None
    friendly: str | None
