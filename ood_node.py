import asyncio
from ultralytics import YOLO

from llm import LLM
from config import *


model = YOLO("models/yolo11x.pt")  # 使用 YOLOv11 Extra-Large 模型
small_llm = LLM(init_msg="你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否可能有被困人员。")
large_llm = LLM(model=MODEL_VL72,
                init_msg="你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否可能有被困人员，如果有，请从操作库中选择需要执行的操作(每次只可选择1种操作)。")

async def ood_stages(img_queue: asyncio.Queue, action_queue: asyncio.Queue):
    """观察、判断、决策环节"""
    detected_classes = set(['background'])

    while True:
        image = img_queue.get()
        results = model(image)
        new_detected_classes = set([result.names[int(box.cls[0])] for result in results for box in result.boxes]).add('background')
        if not new_detected_classes <= detected_classes:
            result = results[0].plot()
            answer = await small_llm.call("参考YOLO模型的识别结果判断附近是否可能存在被困人员，请回答'有'或'没有'。", result)
            
            while True:
                if "没有" in answer:
                    detected_classes = new_detected_classes.copy()
                    break
                elif "有" in answer:
                    answer = await large_llm.call("附近是否可能有被困人员，请回答选择需要执行的操作或'没有'。(可执行操作包含:'在附近搜寻被困人员')", image)
                    
                    while True:
                        if "在附近搜寻被困人员" in answer:
                            action_queue.put_nowait("seek")
                        elif "没有" in answer:
                            detected_classes = new_detected_classes.copy()
                            break
                        else:
                            answer = await large_llm.call("格式输出错误，请回答选择需要执行的操作或'没有'。", image)
                else:
                    answer = await small_llm.call("格式输出错误，请回答'有'或'没有'。", result)
            
            small_llm.clear_messages()
