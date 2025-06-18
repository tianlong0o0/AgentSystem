import asyncio
import numpy as np
from typing import Callable

import drone
from action_lib import *


async def check_queue(queue: asyncio.Queue, time: float) -> str | None:
    """
    在限定时间内检查队列是否有输入
    """
    try:
        message = await asyncio.wait_for(queue.get(), timeout=time)
        return message
    except asyncio.TimeoutError:
        return None
    
def match_action(action: str) -> Callable[[drone.Drone], None]:
    """
    根据动作名返回相应的动作函数
    """
    if action == "seek":
        return seek
    else:
        return None
    
async def check_action_status(drone: drone.Drone, action_queue: asyncio.Queue, time: float):
    """
    检查当前无人机行动状态

    Args:
        time:检查所需时间
    """
    action = await check_queue(action_queue, time)
    if action is not None:
        action_func = match_action(action)
        action_func(drone)

def cal_direction(drone_pos: list, target_pos: list, obstacle_diagram: list) -> int:
    """
    计算无人机移动方向(角度)

    Params:
        obstacle_diagram:障碍分布图
    """

    ref_angel = cal_angel_index(np.array(target_pos) - np.array(drone_pos), 1)
    index = int(ref_angel / 5)
    if obstacle_diagram[index] == 0:
        direction = ref_angel
    else:
        count_r = 0
        index_r = index
        for i in range(36):
            count_r += 1
            index_r = 0 if (index_r == 71) else (index_r + 1)
            if obstacle_diagram[index_r] == 0: break
        count_l = 0
        index_l = index
        for i in range(71):
            count_l -= 1
            index_l = 71 if (index_l == 0) else (index_l - 1)
            if obstacle_diagram[index_l] == 0: break
        if (count_r + count_l) == 72: return None
        else:
            direction = (index_r * 5) if (count_r < count_l) else (index_l * 5)

    return direction


def cal_angel_index(pos: list, range: int=5) -> int:
    """
    计算相对原点的离散角度序号

    Params:
        range:每个区间角度范围
    """
    angel = np.degrees(np.arctan2(pos[1], pos[0]) % (2 * np.pi))
    index = int(angel / range)

    return index


def generate_path(drone_pos: list, target_pos: list, obstacle_diagram, step: float=1.0) -> list:
    """
    生成下一路径点NED坐标

    Params:
        step:无人机移动步长
    """
    direction = cal_direction(drone_pos, target_pos, obstacle_diagram)
    angel_rad = np.radians(direction)
    relative_pos = np.array([np.cos(angel_rad), np.sin(angel_rad), 0])
    next_pos = np.array(drone_pos) + relative_pos * step
    next_pos[2] = target_pos[2]

    return list(next_pos)

def generate_od(lidar_data: list, o_range: float=1.5) -> list:
    """
    生成障碍分布图

    Params:
        step:无人机移动步长
    """
    obstacle_diagram = [0] * 72

    for point in lidar_data:
        distance = np.linalg.norm(np.array([point[0], point[1]]))
        if (distance < o_range) and (abs(point[2] - 0.5) < 0.5):
            index = cal_angel_index(point)
            obstacle_diagram[index] += 1
            index_l = index
            index_r = index
            for i in range(18):
                index_l = 71 if (index_l == 0) else (index_l - 1)
                index_r = 0 if (index_r == 71) else (index_r + 1)
                obstacle_diagram[index_l] += 1
                obstacle_diagram[index_r] += 1

    return obstacle_diagram


async def main(drone: drone.Drone, action_queue: asyncio.Queue):
    way_points = [
                  [-3909, -21, -1]
                  ]  
    await drone.move_to_pos([0, 0, -1])

    while True:
        for target_pos in way_points:
            while np.linalg.norm(np.array(target_pos) - np.array(drone.get_pos())) >= 1:
                obstacle_diagram = generate_od(drone.get_lidar_data())
                pos = generate_path(drone.get_pos(), target_pos, obstacle_diagram, 0.7)
                await drone.move_to_pos(pos)
                await check_action_status(drone, action_queue, 0.5)






