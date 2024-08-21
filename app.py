import json
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from core.models import BartModel, ClipModel, Llama31Model, MinillmModel
from service.agent import Agent
from tools.logger import config_logger
from tools.user_register import UserHandler

# init log
logger = config_logger(
    log_name="system.log",
    logger_name="system",
    default_folder="./log",
    write_mode="w",
    level="debug",
)

# Instantiation
SHARE_HOST = "host.docker.internal"
user_handler = UserHandler()
gen_text_model = Llama31Model(host="ollama")
text_emb_model = MinillmModel(host="ollama")
img_emb_model = ClipModel(host=SHARE_HOST)
topics_classifier_model = BartModel(host=SHARE_HOST)

# Extension map
CONTENT_TYPE_MAP = {"image/jpeg": ".jpg", "image/png": ".png"}


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
    img_emb_model._load_model()
    logger.info(
        f"Success init model to Embedding image. model name = '{img_emb_model.model_name}'"
    )
    topics_classifier_model._load_model()
    logger.info(
        f"Success init model to Do topics classifer. model name = '{topics_classifier_model.model_name}'"
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

# init Service
agent = Agent(
    gen_text_model=gen_text_model,
    text_emb_model=text_emb_model,
    img_emb_model=img_emb_model,
    topics_classifier_service=topics_classifier_model,
    topics=topics,
    host=SHARE_HOST,
)
logger.info("Success init Agent")

app = FastAPI(lifespan=lifespan)
# app.mount("/static", StaticFiles(directory="static", html=True), name="static")
# logger.info("Success init webui, Start service!!!")


@app.post("/chat/")
def chat(
    username: str = Form(...),
    department: str = Form(...),
    file: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None),
    friendly: Optional[str] = Form(None),
):
    logger.info(f"user prompt : {prompt}")
    try:
        file_extension = CONTENT_TYPE_MAP[file.content_type]
        file_name = file.filename if file.filename else f"unknown{file_extension}"
        image = {"img": (file_name, file.file, file.content_type)}
    except BaseException:
        image = None

    return StreamingResponse(
        content=agent.chat(
            log=user_handler.get(username=username, department=department),
            prompt=prompt,
            file=image,
            friendly=friendly,
        ),
        media_type="text/plain",
    )


@app.post("/submit/")
def submit(
    username: str,
    department: str,
):
    response = dict()

    try:
        if user_handler.check(username=username, department=department):
            response["message"] = f"User: '{username}' has been registered !"
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
        user_handler.register(username=username, department=department)
    except BaseException as e:
        return Response(
            content=json.dumps({"messages": str(e)}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    response["message"] = (
        f"User '{username}' , Department : '{department}' successfully registered!"
    )
    return Response(
        content=json.dumps(response),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.post("/report/")
def report(username: str, department: str, feedback: str):
    response = dict()

    try:
        if not user_handler.check(username=username, department=department):
            response["message"] = f"User '{username}' has not registered yet."
            return Response(
                content=json.dumps(response),
                status_code=status.HTTP_200_OK,
                media_type="application/json",
            )
    except BaseException as e:
        return Response(
            content=json.dumps({"messages": str(e)}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="applicatio/json",
        )

    try:
        log = user_handler.get(username=username, department=department)
        log.info(f"User feedback : '{feedback}'")
    except BaseException as e:
        return Response(
            content=json.dumps({"messages": str(e)}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    response["message"] = (
        f"User '{username}' , Department : '{department}' successfully send feedback!"
    )
    return Response(
        content=json.dumps(response),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8007, reload=True)
