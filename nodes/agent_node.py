import asyncio
from ultralytics import YOLO
import numpy as np
import cv2
import time

from llm import LLM
from config import *
from utils import check_queue


model = YOLO("models/yolo11x.pt")  # 使用 YOLOv11 Extra-Large 模型

def yolo_fliter(image: np.uint8) -> tuple[np.uint8, list]:
    """
    yolo模型识别并过滤掉结果中过小的框

    Args:
        image:场景图像
    Returns:
        np.uint8:过滤后识别结果图像
        list:过滤后识别结果列表
    """
    image = image.copy()
    results = model(image, verbose=False)
    detected_classes = set()
    MIN_HEIGHT = 120
    MIN_CONFIDENCE = 0.5

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = box.cls[0]
            bbox_height = y2 - y1
            confidence = box.conf[0]

            if bbox_height > MIN_HEIGHT and confidence > MIN_CONFIDENCE:
                detected_classes.add(result.names[class_id])
                label = f"{result.names[int(class_id)]} {confidence:.2f}"
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)               

    return image, detected_classes


async def observe(img_queue: asyncio.Queue, action_queue: asyncio.Queue, feedback_queue: asyncio.Queue):
    """
    识别场景中的异常并判断是否与任务有关
    Args:
        img_queue:相机图像队列
        action_queue:无人机动作队列
        detected_classes:上次检测目标集合
        feedback_queue:反馈信息队列
    """
    image = await img_queue.get()
    _, new_detected_classes = yolo_fliter(image)

    if "person" in new_detected_classes:
        action_queue.put_nowait("moveto")
        feedback = await check_feedback(feedback_queue)
        action_queue.put_nowait("broadcast")
        feedback = await check_feedback(feedback_queue)
        action_queue.put_nowait("drop")
        feedback = await check_feedback(feedback_queue)
        action_queue.put_nowait("console")
        feedback = await check_feedback(feedback_queue)
        action_queue.put_nowait("seek_next")
        feedback = await check_feedback(feedback_queue)
        await asyncio.sleep(10)


async def check_feedback(feedback_queue: asyncio.Queue):
    feedback = None
    while feedback is None:
        feedback = await check_queue(feedback_queue)

    return feedback

async def main(img_queue: asyncio.Queue, action_queue: asyncio.Queue, feedback_queue: asyncio.Queue):
    while True:
        await observe(img_queue, action_queue, feedback_queue)

