import asyncio
from ultralytics import YOLO
import time
import numpy as np
import cv2

from llm import LLM
from config import *
from utils import check_queue


model = YOLO("models/yolo11x.pt")  # 使用 YOLOv11 Extra-Large 模型
small_llm = LLM(init_msg="你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否有被困人员。")
large_llm = LLM(model=MODEL_MAX_VL,
                init_msg="你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否有被困人员，如果有，请从操作库中选择需要执行的操作(每次只可选择1种操作)。")
    
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


async def observe(img_queue: asyncio.Queue, action_queue: asyncio.Queue, detected_classes: set, feedback_queue: asyncio.Queue):
    """
    识别场景中的异常并判断是否与任务有关
    Args:
        img_queue:相机图像队列
        action_queue:无人机动作队列
        detected_classes:上次检测目标集合
        feedback_queue:反馈信息队列
    """
    image = await img_queue.get()
    result, new_detected_classes = yolo_fliter(image)

    new_detected_classes.add('background')
    if not new_detected_classes <= detected_classes:
        answer = small_llm.call("参考YOLO模型的识别结果判断附近是否可能有被困人员，请回答'有'或'没有'。", result)
        
        while True:
            if "没有" in answer:
                break
            elif "有" in answer:
                # cv2.imshow("Detection Result", image)  # 第一个参数是窗口名，第二个是图像
                # cv2.waitKey(0)  # 等待按键
                await make_decision(img_queue, action_queue, feedback_queue)
                break
            else:
                answer = small_llm.call("格式输出错误，请回答'有'或'没有'。", result)

        small_llm.clear_messages()
    
    return new_detected_classes.copy()

async def make_decision(img_queue: asyncio.Queue, action_queue: asyncio.Queue, feedback_queue: asyncio.Queue):
    """
    场景判断并作出相应决策
    Args:
        img_queue:相机图像队列
        action_queue:无人机动作队列
        feedback_queue:反馈信息队列
    """
    image = await img_queue.get()
    answer = large_llm.call("附近是否可能有被困人员，请回答'有'或'没有'。", image)
    
    while True:
        if "没有" in answer:
            large_llm.del_last_message()
            break
        elif "有" in answer:
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
        else:
            answer = large_llm.call("格式输出错误，请回答选择需要执行的操作或'没有'。", image)

async def check_feedback(feedback_queue: asyncio.Queue):
    feedback = None
    while feedback is None:
        feedback = await check_queue(feedback_queue)

    return feedback

async def main(img_queue: asyncio.Queue, action_queue: asyncio.Queue, feedback_queue: asyncio.Queue):
    detected_classes = set(['background'])
    while True:
        detected_classes = await observe(img_queue, action_queue, detected_classes, feedback_queue)

