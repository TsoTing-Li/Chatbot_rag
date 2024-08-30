from pathlib import Path
from typing import List

import aiofiles
from fastapi import UploadFile

import app


async def async_write_file(path: Path, file: UploadFile, chunk_size: int) -> tuple:
    async with aiofiles.open(path, "wb") as f:
        while chunk := await file.read(chunk_size):
            await f.write(chunk)


async def async_write_multi_files(
    files: List[UploadFile], save_dir: Path, chunk_size: int
):
    if not app.tasks_status.get(save_dir.name):
        app.tasks_status.update({save_dir.name: dict()})
    app.tasks_status[save_dir.name].update({"task": False})
    for file in files:
        path = Path(save_dir) / file.filename
        try:
            await async_write_file(path=path, file=file, chunk_size=chunk_size)
            app.tasks_status[save_dir.name].update({"filename": f"{file.filename}"})
            app.tasks_status[save_dir.name].update({"is_saved": True})
        except BaseException:
            app.tasks_status[save_dir.name].update({"filename": f"{file.filename}"})
            app.tasks_status[save_dir.name].update({"is_saved": False})

    app.tasks_status[save_dir.name].update({"task": True})
