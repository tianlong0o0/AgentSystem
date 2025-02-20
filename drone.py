import airsim
import time
import os
import asyncio
import numpy as np


# 设定保存图片的目录
save_dir = os.path.join(os.getcwd(), 'images') 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 设定飞行参数
side_length = 15
speed = 3
flight_time = side_length / speed
height = -5
edges = [
    (0, speed, 90),     # 东向: vy=3, 偏航90度
    (speed, 0, 0),      # 北向: vx=3, 偏航0度
    (0, -speed, -90),   # 西向: vy=-3, 偏航-90度
    (-speed, 0, 180)    # 南向: vx=-3, 偏航180度
]

async def take_photos(client):
    while True:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        if img_rgb is not None:
            filename = f"{int(time.time()*1000)}.png"
            filepath = os.path.join(save_dir, filename)
            airsim.write_png(filepath, img_rgb)
            print(f"图片已保存: {filepath}")
        
        await asyncio.sleep(0.5)

async def flight(client):
    for vx, vy, yaw in edges:
        # 设置偏航模式（角度模式）
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw)
        # 执行速度控制
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
        await asyncio.sleep(2)

async def main(client):
    await asyncio.gather(take_photos(client), flight(client))


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

    asyncio.run(main(client))
    time.sleep(2)

    # 降落
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)