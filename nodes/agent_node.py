import asyncio
from ultralytics import YOLO
import time
import numpy as np

from llm import LLM
from config import *


model = YOLO("models/yolo11x.pt")  # 使用 YOLOv11 Extra-Large 模型
small_llm = LLM(init_msg="你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否有被困人员。")
large_llm = LLM(model=MODEL_VL72,
                init_msg="你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否有被困人员，如果有，请从操作库中选择需要执行的操作(每次只可选择1种操作)。")
    

def observe(image: np.uint8, action_queue: asyncio.Queue, detected_classes: set):
    """
    识别场景中的异常并判断是否与任务有关
    Args:
        image:场景图像
        action_queue:无人机动作队列
        detected_classes:上次检测目标集合
    """
    results = model(image)
    new_detected_classes = set([result.names[int(box.cls[0])] for result in results for box in result.boxes])
    new_detected_classes.add('background')
    if not new_detected_classes <= detected_classes:
        result = results[0].plot()
        answer = small_llm.call("参考YOLO模型的识别结果判断附近是否可能有被困人员，请回答'有'或'没有'。", result)
        
        while True:
            if "没有" in answer:
                break
            elif "有" in answer:
                make_decision(image, action_queue)
                break
            else:
                answer = small_llm.call("格式输出错误，请回答'有'或'没有'。", result)

        small_llm.clear_messages()
        return new_detected_classes.copy()
    
    return detected_classes.copy()

def make_decision(image: np.uint8, action_queue: asyncio.Queue):
    """
    场景判断并作出相应决策
    Args:
        image:场景图像
        action_queue:无人机动作队列
    """
    answer = large_llm.call("附近是否可能有被困人员，请回答'有'或'没有'。", image)
    
    while True:
        if "没有" in answer:
            break
        elif "有" in answer:
            answer = large_llm.call("请选择需要执行的操作。(可执行操作包含:'在附近搜寻被困人员','通知总部找到被困人员','投放紧急救援物资','安抚被困人员')", image)
            print(answer)
            image.show()
            if "在附近搜寻被困人员" in answer:
                action_queue.put_nowait("seek")
            time.sleep(1000)
            break
        else:
            answer = large_llm.call("格式输出错误，请回答选择需要执行的操作或'没有'。", image)

async def main(img_queue: asyncio.Queue, action_queue: asyncio.Queue):
    while True:
        detected_classes = set(['background'])
        image = await img_queue.get()
        detected_classes = observe(image, action_queue, detected_classes)

