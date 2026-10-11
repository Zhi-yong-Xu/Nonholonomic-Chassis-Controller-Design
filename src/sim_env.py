"""MuJoCo 仿真环境封装（含 MJCF 参数提取）"""
import numpy as np
import mujoco

from .utils import quat_to_yaw, extract_husky_params, validate_params


class HuskySim:
    """MuJoCo 仿真环境封装，负责模型加载、初态设置、状态读写"""

    WHEEL_JOINTS = ["front_left_wheel", "front_right_wheel",
                    "rear_left_wheel",  "rear_right_wheel"]

    def __init__(self, cfg: dict):
        self.model = mujoco.MjModel.from_xml_path(cfg["model"]["xml_path"])
        self.model.opt.timestep = cfg["model"]["timestep"]
        self.data = mujoco.MjData(self.model)
        self.cfg = cfg

        # ---- 从 MJCF 自动提取运动学参数（唯一参数源）----
        self.params = extract_husky_params(self.model)

        # ---- 与 YAML 参考值交叉校验（可选）----
        validate_params(self.params, cfg)

        # ---- 构造差速运动学映射矩阵 Omega ----
        r = self.params["wheel_radius"]
        a = self.params["half_track"]
        self.Omega = np.array([[1.0 / r,  a / r],
                               [1.0 / r, -a / r]])   # [右轮, 左轮] = Omega @ [v, w]

        # ---- 初态设置 ----
        self._set_init_state(cfg["init_state"])

    def _set_init_state(self, s: dict):
        d = self.data
        d.qpos[0] = s["x"]
        d.qpos[1] = s["y"]
        d.qpos[2] = 0.14228 + s["z_offset"]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for name in self.WHEEL_JOINTS:
            d.joint(name).qpos[0] = 0.0
            d.joint(name).qvel[0] = 0.0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(s["settle_steps"])):
            mujoco.mj_step(self.model, self.data)
        d.qpos[0] = s["x"]
        d.qpos[1] = s["y"]
        mujoco.mj_forward(self.model, self.data)

    # ---- 状态读取 ----
    def get_pose(self) -> np.ndarray:
        d = self.data
        return np.array([d.qpos[0], d.qpos[1],
                         quat_to_yaw(d.qpos[3:7])])

    def get_wheel_speeds(self) -> np.ndarray:
        """返回 [右轮平均角速度, 左轮平均角速度]"""
        d = self.data
        phi_FL = d.joint("front_left_wheel").qvel[0]
        phi_FR = d.joint("front_right_wheel").qvel[0]
        phi_RL = d.joint("rear_left_wheel").qvel[0]
        phi_RR = d.joint("rear_right_wheel").qvel[0]
        phi_L = 0.5 * (phi_FL + phi_RL)
        phi_R = 0.5 * (phi_FR + phi_RR)
        return np.array([phi_R, phi_L])

    # ---- 指令下发 ----
    def apply_wheel_torques(self, tau_R: float, tau_L: float):
        d = self.data
        d.actuator("front_right_wheel_motor").ctrl[0] = tau_R
        d.actuator("rear_right_wheel_motor").ctrl[0]  = tau_R
        d.actuator("front_left_wheel_motor").ctrl[0]  = tau_L
        d.actuator("rear_left_wheel_motor").ctrl[0]   = tau_L

    # ---- 步进 ----
    def step(self):
        mujoco.mj_step(self.model, self.data)

    @property
    def timestep(self) -> float:
        return self.model.opt.timestep
