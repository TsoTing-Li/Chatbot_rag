import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse

import schema
from core.models import BartModel, Llama31Model, MinillmModel
from service.agent import Agent
from tools.connect_handler import ConnectHandler
from tools.logger import config_logger
from tools.user_register import UserHandler
from utils import async_write_multi_files

# init log
logger = config_logger(
    log_name="system.log",
    logger_name="system",
    default_folder="./log",
    write_mode="w",
    level="debug",
)

# Instantiation
user_handler = UserHandler()
connect_handler = ConnectHandler()
gen_text_model = Llama31Model(host=connect_handler.OLLAMA_HOST)
text_emb_model = MinillmModel(host=connect_handler.OLLAMA_HOST)
topics_classifier_model = BartModel(host=connect_handler.BART_HOST)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init model
    gen_text_model._load_model()
    logger.info(
        f"Success init model to Gen text. model name = '{gen_text_model.model_name}'"
    )
    text_emb_model._load_model()
    logger.info(
        f"Success init model to Embedding text. model name = '{text_emb_model.model_name}'"
    )
    topics_classifier_model._load_model()
    logger.info(
        f"Success init model to Topics classifier. model name = '{topics_classifier_model.model_name}'"
    )

    yield

    gen_text_model._release_model()
    text_emb_model._release_model()


topics = [
    "Product Information",
    "Pricing and Promotions",
    "Purchasing and Orders",
    "After-sales Service",
    "Company Information",
]
logger.info(f"Setting topics.'{topics}'")

tasks_status = dict()
logger.info("Success create 'tasks_status' dict.")

CHUNK_SIZE = 1024 * 1024 * 5
logger.info(f"Chunk size: {CHUNK_SIZE}")

SAVE_PATH = "upload_pdf"
logger.info(f"Save path: {SAVE_PATH}")

# init Service
agent = Agent(
    gen_text_model=gen_text_model,
    text_emb_model=text_emb_model,
    topics_classifier_service=topics_classifier_model,
    topics=topics,
)
logger.info("Success init Agent")

app = FastAPI(lifespan=lifespan)


@app.post("/chat/", tags=["Chat"])
def chat(
    username: str = Form(...),
    department: str = Form(...),
    prompt: Optional[str] = Form(None),
    friendly: Optional[str] = Form(None),
):
    request_data = schema.PostChat(
        username=username, department=department, prompt=prompt, friendly=friendly
    )
    logger.info(f"user prompt : {prompt}")

    return StreamingResponse(
        content=agent.chat(
            log=user_handler.get(
                username=request_data.username, department=request_data.department
            ),
            prompt=request_data.prompt,
            friendly=request_data.friendly,
        ),
        media_type="text/plain",
    )


@app.post("/submit/", tags=["Submit"])
def submit(request_data: schema.PostSubmit):
    response = dict()

    try:
        if user_handler.check(
            username=request_data.username, department=request_data.department
        ):
            response["message"] = (
                f"User: '{request_data.username}' has been registered !"
            )
            return Response(
                content=json.dumps(response),
                status_code=status.HTTP_200_OK,
                media_type="application/json",
            )
    except BaseException as e:
        print(f"Error: {str(e)}")
        return Response(
            content=json.dumps({"messages": str(e)}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    try:
        user_handler.register(
            username=request_data.username, department=request_data.department
        )
    except BaseException as e:
        return Response(
            content=json.dumps({"messages": str(e)}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    response["message"] = (
        f"User '{request_data.username}' , Department : '{request_data.department}' successfully registered!"
    )
    return Response(
        content=json.dumps(response),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.post("/report/", tags=["Report"])
def report(request_data: schema.PostReport):
    response = dict()

    try:
        if not user_handler.check(
            username=request_data.username, department=request_data.department
        ):
            response["message"] = (
                f"User '{request_data.username}' has not registered yet."
            )
            return Response(
                content=json.dumps(response),
                status_code=status.HTTP_200_OK,
                media_type="application/json",
            )
    except BaseException as e:
        return Response(
            content=json.dumps({"messages": str(e)}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    try:
        log = user_handler.get(
            username=request_data.username, department=request_data.department
        )
        log.info(f"User feedback : '{request_data.feedback}'")
    except BaseException as e:
        return Response(
            content=json.dumps({"messages": str(e)}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    response["message"] = (
        f"User '{request_data.username}' , Department : '{request_data.department}' successfully send feedback!"
    )
    return Response(
        content=json.dumps(response),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.get("/status/", tags=["Status"])
async def get_status(_id: str):
    progress = tasks_status.get(_id)
    if not progress:
        progress = dict()
    return Response(
        content=json.dumps(progress),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.post("/upload/", tags=["Upload"])
async def upload(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    username: str = Form(...),
    department: str = Form(...),
):
    request_data = schema.PostUpload(
        files=files, username=username, department=department
    )

    current_file_path = Path(__file__).resolve().parent
    save_dir = (
        current_file_path
        / SAVE_PATH
        / f"{request_data.department}_{request_data.username}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    background_tasks.add_task(
        async_write_multi_files, request_data.files, save_dir, CHUNK_SIZE
    )

    return Response(
        content=json.dumps({"task_id": f"{save_dir.name}"}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.websocket("/ws/{task_id}")
async def websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    previous_filename = None

    try:
        while True:
            progress = tasks_status.get(task_id)
            if progress is None:
                break
            if progress["filename"] != previous_filename:
                try:
                    await websocket.send_json(progress)
                    previous_filename = progress["filename"]
                except BaseException:
                    break
            if progress["task"] is True:
                break

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print(f"WebSocket connection closed for task: {task_id}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8007, reload=True)
