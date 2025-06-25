import asyncio
import numpy as np
from typing import Callable

import drone
from action_lib import *
from utils import check_queue

    
def match_action(action: str) -> Callable[[drone.Drone, asyncio.Queue], bool]:
    """
    根据动作名返回相应的动作函数
    """
    if action == "seek":
        return seek
    elif action == "moveto":
        return moveto
    elif action == "broadcast":
        return broadcast
    elif action == "drop":
        return drop
    elif action == "console":
        return console
    elif action == "seek_next":
        return seek_next
    else:
        return None
    
async def check_action_status(drone: drone.Drone, action_queue: asyncio.Queue, feedback_queue: asyncio.Queue) -> bool:
    """
    检查当前无人机行动状态并反馈

    Returns:
        bool:是否跳过下一导航点
    """
    signal = False
    action = await check_queue(action_queue)
    if action is not None:
        while True:
            action_func = match_action(action)
            signal = await action_func(drone, feedback_queue)
            if signal: break
            action = await action_queue.get()

    return signal

async def main(drone: drone.Drone, action_queue: asyncio.Queue, feedback_queue: asyncio.Queue):
    way_points = [
                  [-38, 0, -1],
                  [-38, -10, -1],
                  [-56.5, -46, -1],
                  [-66.5, 0.7, -1],
                  [-67, -0.7, -1]
                  ]  
    await drone.move_to_pos([0, 0, -1])
    signal = False

    while True:
        for target_pos in way_points:
            if signal:
                for i in range(10):
                    await drone.move_to_pos_oa(target_pos)

            while np.linalg.norm(np.array(target_pos) - np.array(drone.get_pos())) >= 1:
                await drone.move_to_pos_oa(target_pos)
                signal = await check_action_status(drone, action_queue, feedback_queue)
                if signal: break






