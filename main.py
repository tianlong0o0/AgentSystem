import asyncio

from config import *
import nodes
from drone import Drone
import nodes.agent_node
import nodes.camera_node
import nodes.drone_node


async def main():
    drone = Drone()
    action_queue = asyncio.Queue()
    feedback_queue = asyncio.Queue()
    img_queue = asyncio.Queue(maxsize=1)
    await asyncio.gather(nodes.camera_node.main(drone, img_queue, 0.1),
                         nodes.drone_node.main(drone, action_queue, feedback_queue),
                         nodes.agent_node.main(img_queue, action_queue, feedback_queue))

if __name__ == "__main__":
    asyncio.run(main())


