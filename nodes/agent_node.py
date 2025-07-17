import asyncio
from ultralytics import YOLO
import time
import numpy as np
import cv2

from llm import LLM
from config import *
from utils import check_queue


large_llm = LLM(model=MODEL_MAX_VL,
                init_msg="你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否有被困人员，如果有，请从操作库中选择需要执行的操作(每次只可选择1种操作)。")


async def observe(img_queue: asyncio.Queue, action_queue: asyncio.Queue, feedback_queue: asyncio.Queue):
    """
    识别场景中的异常并判断是否与任务有关
    Args:
        img_queue:相机图像队列
        action_queue:无人机动作队列
        detected_classes:上次检测目标集合
        feedback_queue:反馈信息队列
    """
    while True:
        image = await img_queue.get()
        answer = large_llm.call("参考YOLO模型的识别结果判断附近是否可能有被困人员，请回答'有'或'没有'。", image)
        if "没有" in answer:
            break
        elif "有" in answer:
            # cv2.imshow("Detection Result", image)  # 第一个参数是窗口名，第二个是图像
            # cv2.waitKey(0)  # 等待按键
            await make_decision(img_queue, action_queue, feedback_queue)
            break
        else:
            answer = large_llm.call("格式输出错误，请回答'有'或'没有'。", image)
        large_llm.clear_messages()
        await asyncio.sleep(0.1)


async def make_decision(img_queue: asyncio.Queue, action_queue: asyncio.Queue, feedback_queue: asyncio.Queue):
    """
    场景判断并作出相应决策
    Args:
        img_queue:相机图像队列
        action_queue:无人机动作队列
        feedback_queue:反馈信息队列
    """
    image = await img_queue.get()
    
    while True:
        answer = large_llm.call("请选择需要执行的操作。(可执行操作包含:'移动至被困人员处','在被困人员物品周围搜寻被困人员','通知总部找到被困人员','向被困人员发放紧急救援物资','安抚被困人员','继续寻找其他被困人员')", image)
        while True:
            if "在被困人员物品周围搜寻被困人员" in answer:
                action_queue.put_nowait("seek")
            elif "移动至被困人员处" in answer:
                action_queue.put_nowait("moveto")
            elif "通知总部找到被困人员" in answer:
                action_queue.put_nowait("broadcast")
            elif "向被困人员发放紧急救援物资" in answer:
                action_queue.put_nowait("drop")
            elif "安抚被困人员" in answer:
                action_queue.put_nowait("console")
            elif "继续寻找其他被困人员" in answer:
                action_queue.put_nowait("seek_next")
                feedback = await check_feedback(feedback_queue)
                large_llm.user_put(feedback)
                break
            else:
                answer = large_llm.call("未从回答中识别到可执行操作，请输出正确的需要执行的操作。(可执行操作包含:'移动至被困人员处','在被困人员物品周围搜寻被困人员','通知总部找到被困人员','向被困人员发放紧急救援物资','安抚被困人员','继续寻找其他被困人员')", image)
            
            feedback = await check_feedback(feedback_queue)
            image = await img_queue.get()
            answer = large_llm.call(feedback, image)
        break


async def check_feedback(feedback_queue: asyncio.Queue):
    feedback = None
    while feedback is None:
        feedback = await check_queue(feedback_queue)

    return feedback

async def main(img_queue: asyncio.Queue, action_queue: asyncio.Queue, feedback_queue: asyncio.Queue):
    while True:
        await observe(img_queue, action_queue, feedback_queue)

