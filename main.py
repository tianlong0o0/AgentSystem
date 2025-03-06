import airsim
import time
import asyncio

from config import *
import act_node, camera_node, ood_node


async def main(client):
    action_queue = asyncio.Queue()
    img_queue = asyncio.Queue(maxsize=1)
    await asyncio.gather(camera_node.img_input(client, img_queue, 0.1),
                         act_node.default(client, action_queue),
                         ood_node.ood_stages(img_queue, action_queue))

if __name__ == "__main__":
    # 连接到AirSim模拟器
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # 起飞
    client.takeoffAsync().join()
    client.moveToZAsync(-1, 2).join()
    time.sleep(2)

    # 执行任务
    asyncio.run(main(client))
    time.sleep(2)

    # 降落
    client.moveToZAsync(0, 2).join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)


