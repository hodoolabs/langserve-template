from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes
from app.chain import chain
from app.chat import chain as chat_chain
from app.image_to_text import image_to_text, ImageToTextInput, ImageToTextOutput
from base64 import b64encode

from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
from langchain_core.document_loaders import Blob
from langchain_core.runnables import RunnableLambda
import base64
from langserve import CustomUserType


from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/prompt/playground")


add_routes(app, chain, path="/prompt")


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


add_routes(
    app,
    chat_chain.with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)


# @app.post("/upload-image")
# async def upload_image(
#     file: UploadFile = File(...), prompt: str = "이 이미지에 대해 설명해주세요."
# ):
#     contents = await file.read()
#     base64_image = b64encode(contents).decode("utf-8")
#     input_data = ImageToTextInput(image=base64_image, prompt=prompt)
#     result = image_to_text_chain.invoke(input_data)
#     return result

# 이미지 처리를 위한 RunnableLambda
image_to_text_runnable = RunnableLambda(image_to_text)

add_routes(
    app,
    # image_to_text_chain.with_types(input_type=ImageToTextInput),
    image_to_text_runnable.with_types(
        input_type=ImageToTextInput, output_type=ImageToTextOutput
    ),
    path="/image-to-text",
    # enable_feedback_endpoint=True,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
