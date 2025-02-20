from openai import OpenAI
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
        self.messages = [{"role": "system", "content": init_msg}]

    def _call_llm(self) -> str:
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        completion = client.chat.completions.create(
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

    def call(self, text: str, image=None) -> str:
        self._update_messages(Role.user, text, image)
        output = self._call_llm()
        self._update_messages(Role.assistant, output)

        return output

