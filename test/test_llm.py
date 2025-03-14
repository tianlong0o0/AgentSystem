import pytest
import cv2, asyncio

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


def test_llm_chat_async(init_llm_chat):
    """异步大模型多轮文字调用测试"""
    assert "2" in asyncio.run(init_llm_chat.call_async("1+1等于几"))
    assert "equal" in asyncio.run(init_llm_chat.call_async("用英文回答上个问题"))

def test_llm_vl_async(init_llm_vl):
    """异步大模型多轮视觉调用测试"""
    assert "猫" in asyncio.run(init_llm_vl.call_async("图片里有什么(中文回答)", "https://qcloud.dpfile.com/pc/52biLHyKFbBtEA4iUSAEozWShodbuvI2I68FwiotXY3GrJw0LHxnoC2W1HJHsNeU.jpg"))
    assert "狗" in asyncio.run(init_llm_vl.call_async("图片里有什么(中文回答)", cv2.imread("test/dog.jpg")))

def test_llm_clear_async(init_llm_chat):
    """异步llm类clear_messages方法测试"""
    asyncio.run(init_llm_chat.call_async("1+1等于几"))
    init_llm_chat.clear_messages()
    assert 1 == len(init_llm_chat.messages)

def test_llm_chat(init_llm_chat):
    """大模型多轮文字调用测试"""
    assert "2" in init_llm_chat.call("1+1等于几")
    assert "equal" in init_llm_chat.call("用英文回答上个问题")

def test_llm_vl(init_llm_vl):
    """大模型多轮视觉调用测试"""
    assert "猫" in init_llm_vl.call("图片里有什么(中文回答)", "https://qcloud.dpfile.com/pc/52biLHyKFbBtEA4iUSAEozWShodbuvI2I68FwiotXY3GrJw0LHxnoC2W1HJHsNeU.jpg")
    assert "狗" in init_llm_vl.call("图片里有什么(中文回答)", cv2.imread("test/dog.jpg"))

def test_llm_clear(init_llm_chat):
    """llm类clear_messages方法测试"""
    init_llm_chat.call("1+1等于几")
    init_llm_chat.clear_messages()
    assert 1 == len(init_llm_chat.messages)