import airsim, math, asyncio
import numpy as np


def cal_angle(pos_now: list, pos_next: list) -> float:
    """
    计算两个位置坐标的方位角

    Returns:
        角度计算结果
    """

    return math.degrees(math.atan2(pos_next[1]-pos_now[1], pos_next[0]-pos_now[0]))


class Drone:
    def __init__(self):
        # 连接到AirSim模拟器
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    async def move_to_pos(self, pos: list, velocity: float=1.0):
        """
        移动至指定坐标(始终朝向移动方向)
        """
        # 获取当前位置
        pos_now = self.get_pos()
        distance = np.linalg.norm(np.array(pos) - np.array(pos_now))

        # 计算朝向并移动
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=cal_angle(pos_now, pos))
        self.client.moveToPositionAsync(pos[0], pos[1], pos[2], velocity, yaw_mode=yaw_mode)

        # 异步
        await asyncio.sleep(distance / velocity)
        self.hover()

    def get_pos(self) -> list:
        """
        获取无人机当前位置
        """
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        pos = [position.x_val, position.y_val, position.z_val - 0.5]

        return pos

    def take_off(self):
        """
        无人机起飞至默认高度
        """
        self.client.takeoffAsync().join()

    def land(self):
        """
        无人机降落并脱离控制
        """
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
    
    def hover(self):
        """
        无人机悬停
        """
        self.client.hoverAsync()

    def take_photos(self) -> np.uint8:
        """
        使用无人机前置相机拍照
        """
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)

        return img_rgb
    
    def get_lidar_data(self) -> np.ndarray:
        """
        获取激光雷达点云数据

        Returns:
            numpy二维数组形式的点云数据(每个元素为一个点的NED坐标)
        """
        data = self.client.getLidarData()
        point_cloud = np.array(data.point_cloud)
        point_cloud = point_cloud.reshape(-1, 3)

        return point_cloud