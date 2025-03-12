import asyncio
import numpy as np
import scipy.interpolate
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
        time:队列是否有输入检查时间
    """
    action = await check_queue(action_queue, time)
    if action is not None:
        action_func = match_action(action)
        action_func(drone)

def generate_path(waypoints: list, step: float=1.0) -> list:
    """
    利用已知路径点拓展路径至要求步长

    Returns:
        拓展后的路径列表
    """
    waypoints = np.array(waypoints)
    path_length_e = np.sum(np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1))) # 估计路径总长度
    sum_points = int(path_length_e / step)
    x, y , z= waypoints.T

    # 样条插值
    t = np.arange(len(x))
    t_new = np.linspace(0, len(x)-1, sum_points)
    spl = scipy.interpolate.make_interp_spline(t, np.c_[x, y, z], k=3)

    return list(spl(t_new))

async def main(drone: drone.Drone, action_queue: asyncio.Queue):
    way_points = [[0, 0, -0.5],
                  [0, 15, -0.5],
                  [15, 15, -0.5],
                  [15, 0, -0.5]]
    path = generate_path(way_points)
    
    while True:
        for pos in path:
            await drone.move_to_pos(pos)
            await check_action_status(drone, action_queue, 0.5)






