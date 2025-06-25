import asyncio
import numpy as np
import math

async def check_queue(queue: asyncio.Queue, time: float=0.5) -> str | None:
    """
    在限定时间内检查队列是否有输入
    """
    try:
        message = await asyncio.wait_for(queue.get(), timeout=time)
        return message
    except asyncio.TimeoutError:
        return None

def cal_pos(pos: list, direction: float, distance: float):
    """
    根据角度，相对距离计算坐标
    """
    
    angel_rad = np.radians(direction)
    relative_pos = np.array([np.cos(angel_rad), np.sin(angel_rad), 0])
    next_pos = np.array(pos) + relative_pos * distance
    next_pos[2] = -1

    return next_pos

def cal_angle(pos_now: list, pos_next: list) -> float:
    """
    计算两个位置坐标的方位角

    Returns:
        角度计算结果
    """

    return math.degrees(math.atan2(pos_next[1]-pos_now[1], pos_next[0]-pos_now[0]))

def cal_angel_index(pos: list, range: int=5) -> int:
    """
    计算相对原点的离散角度序号

    Args:
        range:每个区间角度范围
    """
    angel = cal_angle([0, 0], [pos[0], pos[1]]) % (360)
    index = int(angel / range)

    return index