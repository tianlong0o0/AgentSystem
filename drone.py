import airsim, math, asyncio
import numpy as np

from utils import cal_angle, cal_angel_index, cal_pos


class Drone:
    def __init__(self):
        # 连接到AirSim模拟器
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.obstacle_diagram = [0] * 72 # 障碍分布图

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

    def get_facing(self) -> float:
        """
        获取无人机当前朝向
        """
        state = self.client.getMultirotorState().kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(state)
        yaw = math.degrees(yaw)

        return yaw

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
    
    def _generate_od(self, o_range: float=1.2):
        """
        生成障碍分布图

        Args:
            o_range:障碍物检测范围
        """
        obstacle_diagram = [0] * 72
        lidar_data = self.get_lidar_data()

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

        self.obstacle_diagram = obstacle_diagram

    async def move_to_pos_oa(self, pos: list):
        """
        向指定坐标方向移动一步(带避障)
        """
        pos = self._generate_path(pos, 0.7)
        await self.move_to_pos(pos)

    def _generate_path(self, target_pos: list, step: float=1.0) -> list:
        """
        生成下一路径点NED坐标

        Args:
            step:无人机移动步长
        """
        drone_pos = self.get_pos()
        direction = self._cal_direction(target_pos)
        next_pos = cal_pos(drone_pos, direction, step)

        return list(next_pos)
    
    def _cal_direction(self, target_pos: list) -> int:
        """
        计算无人机移动方向(角度)
        """
        self._generate_od()
        drone_pos = self.get_pos()
        ref_angel = cal_angel_index(np.array(target_pos) - np.array(drone_pos), 1)
        
        index = int(ref_angel / 5)
        if self.obstacle_diagram[index] == 0:
            direction = ref_angel
        else:
            count_r = 0
            index_r = index
            for i in range(36):
                count_r += 1
                index_r = 0 if (index_r == 71) else (index_r + 1)
                if self.obstacle_diagram[index_r] == 0: break
            count_l = 0
            index_l = index
            for i in range(71):
                count_l -= 1
                index_l = 71 if (index_l == 0) else (index_l - 1)
                if self.obstacle_diagram[index_l] == 0: break
            if (count_r + count_l) == 72: return None
            else:
                direction = (index_r * 5) if (count_r < count_l) else (index_l * 5)

        return direction