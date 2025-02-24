from ultralytics import YOLO
import cv2
import airsim
import time
import os
import asyncio
import numpy as np

from llm import LLM
from config import *


async def use_defaults(client):
    # 设定飞行参数
    side_length =14
    speed = 2
    flight_time = 1 / speed
    height = -5
    edges = [
        (0, speed, 90),     # 东向: vy=3, 偏航90度
        (speed, 0, 0),      # 北向: vx=3, 偏航0度
        (0, -speed, -90),   # 西向: vy=-3, 偏航-90度
        (-speed, 0, 180)    # 南向: vx=-3, 偏航180度
    ]

    while True:
        for vx, vy, yaw in edges:
            # 设置偏航模式（角度模式）
            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw)
            # 执行速度控制
            for i in range(side_length):
                client.moveByVelocityZAsync(
                    vx,            # X轴速度（北方向）
                    vy,            # Y轴速度（东方向）
                    height,        # 固定高度
                    flight_time,
                    yaw_mode=yaw_mode
                )
                await asyncio.sleep(flight_time)
            # 悬停稳定
            client.hoverAsync()
            await asyncio.sleep(1)

def take_photos(client) -> np.uint8:
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)

    return img_rgb

async def recognize_img(client):
    model = YOLO("models/yolo11x.pt")  # 使用 YOLOv11 Extra-Large 模型
    small_llm = LLM(init_msg="你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否可能有被困人员。")
    large_llm = LLM(model=MODEL_MAX_VL,
                    init_msg="你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否可能有被困人员，如果有，请从操作库中选择需要执行的操作(每次只可选择1种操作)。")

    detected_classes = set(['background'])
    while True:
        image = take_photos(client)
        results = model(image)
        new_detected_classes = [result.names[int(box.cls[0])] for result in results for box in result.boxes]
        new_detected_classes = set(new_detected_classes)
        new_detected_classes.add('background')
        if not new_detected_classes <= detected_classes:
            result = results[0].plot()
            answer = small_llm.call("附近是否可能有被困人员，请回答'有'或'没有'。", result)
            print(answer)
            while True:
                if "没有" in answer:
                    detected_classes = new_detected_classes.copy()
                    break
                elif "有" in answer:
                    answer = large_llm.call("附近是否可能有被困人员，请回答选择需要执行的操作或'没有'。(可执行操作包含:'移动至附近')", result)
                    results[0].show()
                    print(answer)
                    client.hoverAsync()
                    time.sleep(1000)
                else:
                    answer = small_llm.call("格式输出错误，请回答'有'或'没有'。", result)
            small_llm.clear_messages()

        await asyncio.sleep(1)

async def main(client):
    await asyncio.gather(recognize_img(client), use_defaults(client))

if __name__ == "__main__":
    # 连接到AirSim模拟器
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # 起飞
    client.takeoffAsync().join()
    client.moveToZAsync(-5, 2).join()
    time.sleep(2)

    # 执行任务
    asyncio.run(main(client))
    time.sleep(2)

    # 降落
    client.moveToZAsync(0, 2).join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)


