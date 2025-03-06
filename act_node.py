import asyncio
import airsim
import time
from typing import Callable


# 设定飞行参数
SIDE = 10
SPEED = 2
TIME = 1 / SPEED
HEIGHT = -1
EDGES = [
    (0, SPEED, 90),     # 东向: vy=3, 偏航90度
    (SPEED, 0, 0),      # 北向: vx=3, 偏航0度
    (0, -SPEED, -90),   # 西向: vy=-3, 偏航-90度
    (-SPEED, 0, 180)    # 南向: vx=-3, 偏航180度
]

async def check_queue(queue: asyncio.Queue, time: float) -> str | None:
    try:
        message = await asyncio.wait_for(queue.get(), timeout=time)
        return message
    except asyncio.TimeoutError:
        return None
    
def match_action(action: str) -> Callable[[airsim.MultirotorClient], None]:
    if action == "seek":
        return seek
    else:
        return None
    
async def check_action_status(client: airsim.MultirotorClient, queue: asyncio.Queue, time: float):
    action = await check_queue(queue, time)
    if action is not None:
        action_func = match_action(action)
        action_func(client)

async def default(client: airsim.MultirotorClient, queue: asyncio.Queue):
    while True:
        for vx, vy, yaw in EDGES:
            # 设置偏航模式（角度模式）
            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw)
            # 执行速度控制
            for i in range(SIDE):
                client.moveByVelocityZAsync(
                    vx,            # X轴速度（北方向）
                    vy,            # Y轴速度（东方向）
                    HEIGHT,        # 固定高度
                    TIME,
                    yaw_mode=yaw_mode
                )
                await check_action_status(client, queue, 0.5)
            # 悬停稳定
            client.hoverAsync()
            await check_action_status(client, queue, 0.5)

def seek(client: airsim.MultirotorClient):
    yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
    client.moveByVelocityZAsync(
                    SPEED,            # X轴速度（北方向）
                    0,            # Y轴速度（东方向）
                    HEIGHT,        # 固定高度
                    TIME,
                    yaw_mode=yaw_mode
                )
    time.sleep(1000)