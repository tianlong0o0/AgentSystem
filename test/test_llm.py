import pytest
import cv2

from llm import LLM
from config import *


@pytest.fixture
def init_llm_chat():
    llm_chat = LLM(model=MODEL_MAX)
    yield llm_chat
    del llm_chat

@pytest.fixture()
def init_llm_vl():
    llm_vl = LLM(model=MODEL_MAX_VL)
    yield llm_vl
    del llm_vl


def test_llm_chat(init_llm_chat):
    """大模型多轮文字调用测试"""
    assert "2" in init_llm_chat.call("1+1等于几")
    assert "equal" in init_llm_chat.call("用英文回答上个问题")

def test_llm_vl(init_llm_vl):
    """大模型多轮视觉调用测试"""
    assert "猫" in init_llm_vl.call("图片里有什么(中文回答)", "https://qcloud.dpfile.com/pc/52biLHyKFbBtEA4iUSAEozWShodbuvI2I68FwiotXY3GrJw0LHxnoC2W1HJHsNeU.jpg")
    assert "狗" in init_llm_vl.call("图片里有什么(中文回答)", cv2.imread("test/dog.jpg"))