from openai import AsyncOpenAI
from config import *
from enum import Enum, unique
import numpy as np
from PIL import Image
import io
import base64


@unique
class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


def encode_image(img_rgb: np.uint8) -> str:
    image = Image.fromarray(np.uint8(img_rgb))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str


class LLM:
    def __init__(self, model: str=MODEL_VL3, api_key: str=API_KEY, base_url: str=BASE_URL, init_msg: str="你是一个人工智能助手。"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.init_msg = init_msg
        self.messages = [{"role": "system", "content": init_msg}]

    async def _call_llm(self) -> str:
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        completion = await client.chat.completions.create(
            model=self.model,
            messages=self.messages
            )
        
        return completion.choices[0].message.content
    
    def _update_messages(self, role: Role, text: str, image=None):
        if image is None:
            new_message = {"role": role.value, "content": text}
        elif isinstance(image, str):
            new_message = {"role": role.value, "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url",
                        "image_url": {"url": image}}
                        ]}
        elif isinstance(image, np.ndarray):
            new_message = {"role": role.value, "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(image)}"}}
                        ]}

        self.messages.append(new_message)

    async def call(self, text: str, image=None) -> str:
        """可接受格式为url或np.uint8的image"""
        self._update_messages(Role.user, text, image)
        output = await self._call_llm()
        self._update_messages(Role.assistant, output)

        return output
    
    def clear_messages(self):
        self.messages = [{"role": "system", "content": self.init_msg}]

