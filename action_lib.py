import time, math
import numpy as np
import asyncio
from ultralytics import YOLO

import drone
from utils import cal_pos
from nodes.agent_node import yolo_fliter


async def seek(drone: drone.Drone, feedback_queue: asyncio.Queue) -> bool:
    """
    在附近搜寻被困人员

    Returns:
        bool:是否跳过下一导航点
    """
    
    print("在附近搜寻被困人员")
    drone_pos = drone.get_pos()
    waypoints = [
        [drone_pos[0] + 15, drone_pos[1] - 15, -1],
        [drone_pos[0] - 15, drone_pos[1] - 15, -1],
        [drone_pos[0] - 15, drone_pos[1] + 15, -1],
        [drone_pos[0] + 15, drone_pos[1] + 15, -1],
        [drone_pos[0], drone_pos[1] + 15, -1]
    ]
    for target_pos in waypoints:
        while np.linalg.norm(np.array(target_pos) - np.array(drone.get_pos())) >= 1:
            await drone.move_to_pos_oa(target_pos)
            _, detected_classes = yolo_fliter(drone.take_photos())
            if "person" in detected_classes:
                feedback_queue.put_nowait("已在视野内发现被困人员，请选择下一个需要执行的操作")
                return False

    feedback_queue.put_nowait("未发现附近有被困人员，请选择下一个需要执行的操作")
    
    return False

async def moveto(drone: drone.Drone, feedback_queue: asyncio.Queue) -> bool:
    """
    移动至被困人员处

    Returns:
        bool:是否跳过下一导航点
    """
    print("移动至被困人员处")

    direction = drone.get_facing()
    pos = cal_pos(drone.get_pos(), direction, 100)
    for i in range(10):
        await drone.move_to_pos_oa(pos)

    model = YOLO("models/yolo11x.pt")  # 使用 YOLOv11 Extra-Large 模型
    MIN_HEIGHT = 400
    direction = drone.get_facing()
    drone_pos = drone.get_pos()
    pos = cal_pos(drone.get_pos(), direction, 1.0)
    count = 0
    cal_singal = True

    _, detected_classes = yolo_fliter(drone.take_photos())
    if 'person' not in detected_classes:
        feedback_queue.put_nowait("未在视野内发现被困人员，请确认并再次选择需要执行的操作")
        return False

    while True:
        image = drone.take_photos()
        results = model(image, verbose=False)
        count += 1
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                if class_name != 'person': continue

                count = 0
                x1, y1, x2, y2 = box.xyxy[0]           
                bbox_height = y2 - y1

                if cal_singal:
                    cal_singal = False
                    scale = 1.8 / bbox_height
                    horizontal_offset = int((x1 + x2 - 1920) / 2)
                    relative_yaw = horizontal_offset / 1920 * 90 * 1.25
                    direction = drone.get_facing() + relative_yaw
                    distance = abs(horizontal_offset * scale / math.sin(math.radians(relative_yaw)))

                    pos = cal_pos(drone.get_pos(), direction, distance)

                if bbox_height > MIN_HEIGHT:
                    feedback_queue.put_nowait("已成功移动至被困人员处，请选择下一个需要执行的操作")
                    return False              
        
        await drone.move_to_pos_oa(pos)
        if count == 3:
            feedback_queue.put_nowait("未发现被困人员，请确认并选择需要执行的操作")
            break

    return False

async def broadcast(drone: drone.Drone, feedback_queue: asyncio.Queue) -> bool:
    """
    通知总部找到被困人员

    Returns:
        bool:是否跳过下一导航点
    """
    print(f"无人机于{str(drone.get_pos())}找到被困人员")
    feedback_queue.put_nowait("已成功通知总部，请选择下一个需要执行的操作")

    return False

async def drop(drone: drone.Drone, feedback_queue: asyncio.Queue) -> bool:
    """
    向被困人员发放紧急救援物资

    Returns:
        bool:是否跳过下一导航点
    """
    print("投放紧急救援物资")
    feedback_queue.put_nowait("已成功投放紧急救援物资，请选择下一个需要执行的操作")

    return False

async def console(drone: drone.Drone, feedback_queue: asyncio.Queue) -> bool:
    """
    安抚被困人员

    Returns:
        bool:是否跳过下一导航点
    """
    print("安抚被困人员")
    feedback_queue.put_nowait("已成功安抚被困人员，请选择下一个需要执行的操作")

    return False

async def seek_next(drone: drone.Drone, feedback_queue: asyncio.Queue) -> bool:
    """
    继续寻找其他被困人员

    Returns:
        bool:是否跳过下一导航点
    """
    print("继续寻找其他被困人员")
    feedback_queue.put_nowait("该被困人员已获救，继续搜寻其他被困人员")

    return True