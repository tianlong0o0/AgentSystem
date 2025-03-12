import drone
import asyncio


async def main(drone: drone.Drone, img_queue: asyncio.Queue, refresh_time: float):
    while True:
        img_rgb = drone.take_photos()
        try:
            img_queue.put_nowait(img_rgb)
        except asyncio.QueueFull:
            try:
                img_queue.get_nowait()
            except img_queue.Empty:
                print("Unexpected state: img_queue was reported as full but was empty.")
            
            img_queue.put_nowait(img_rgb)
        await asyncio.sleep(refresh_time)