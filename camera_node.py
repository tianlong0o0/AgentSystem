import numpy as np
import airsim
import asyncio


def take_photos(client: airsim.MultirotorClient) -> np.uint8:
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)

    return img_rgb

async def img_input(client: airsim.MultirotorClient, img_queue: asyncio.Queue, refresh_time: float):
    while True:
        img_rgb = take_photos(client)
        try:
            img_queue.put_nowait(img_rgb)
        except asyncio.QueueFull:
            try:
                img_queue.get_nowait()
            except img_queue.Empty:
                print("Unexpected state: img_queue was reported as full but was empty.")
            
            img_queue.put_nowait(img_rgb)
        asyncio.sleep(refresh_time)