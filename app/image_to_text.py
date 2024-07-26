from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnablePassthrough
from langserve import CustomUserType
import base64


class ImageToTextInput(CustomUserType):
    image: str = Field(
        ...,
        description="Base64로 인코딩된 이미지",
        extra={"widget": {"type": "base64file"}},
    )
    prompt: str = Field(
        default="이 이미지에 대해 설명해주세요.",
        description="이미지 분석을 위한 프롬프트",
    )


class ImageToTextOutput(CustomUserType):
    description: str = Field(..., description="이미지에 대한 설명")


def image_to_text(input_data: ImageToTextInput) -> ImageToTextOutput:
    image_data = input_data.image
    if not image_data.startswith("data:image"):
        image_data = f"data:image/jpeg;base64,{image_data}"

    chat = ChatOpenAI(model="gpt-4o", max_tokens=1024)
    message = HumanMessage(
        content=[
            {"type": "text", "text": input_data.prompt},
            {"type": "image_url", "image_url": {"url": image_data}},
        ]
    )
    # print("message: ", message)
    result = chat.invoke([message])
    print("result: ", result)
    return ImageToTextOutput(description=result.content)


image_to_text_chain = RunnablePassthrough(image_to_text)
