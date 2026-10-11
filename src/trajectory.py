"""
参考轨迹生成模块
"""
import numpy as np


# ============================================================
# 基类
# ============================================================
class BaseTrajectory:
    """参考轨迹基类，子类必须实现 _pose_raw / _vel_raw"""

    # 子类需覆盖：一个完整周期 [s]；开放轨迹设为 None
    period: float = None

    # ---- 子类实现的核心计算 ----
    def _pose_raw(self, t: float) -> np.ndarray:
        """返回 [x, y]"""
        raise NotImplementedError

    def _vel_raw(self, t: float) -> np.ndarray:
        """返回世界系线速度 [vx, vy]"""
        raise NotImplementedError

    # ---- 统一对外接口 ----
    def pose(self, t: float) -> np.ndarray:
        """t 时刻参考位姿 [x, y, yaw]"""
        p = self._pose_raw(t)
        v = self._vel_raw(t)
        yaw = np.arctan2(v[1], v[0])
        return np.array([p[0], p[1], yaw])

    def velocity(self, t: float) -> np.ndarray:
        """t 时刻参考点世界系线速度 [vx, vy]"""
        return self._vel_raw(t)

    def vis_points(self, n: int = 241, z: float = 0.02) -> list:
        """沿轨迹采样可视化点。闭合轨迹取一个周期，开放轨迹取 [0, T_vis]"""
        if self.period is not None:
            ts = np.linspace(0.0, self.period, n, endpoint=False)
        else:
            ts = np.linspace(0.0, self.T_vis, n)
        return [np.array([*self._pose_raw(t)[:2], z]) for t in ts]

    def initial_pos(self) -> np.ndarray:
        """t=0 时刻参考点位置（FTDSO 初始化用）"""
        return self._pose_raw(0.0)[:2].copy()


# ============================================================
# 圆形轨迹
# ============================================================
class CircleTrajectory(BaseTrajectory):

    def __init__(self, radius: float, omega: float):
        self.R = radius
        self.w = omega
        self.period = 2.0 * np.pi / omega

    def _pose_raw(self, t):
        return np.array([self.R * np.cos(self.w * t),
                         self.R * np.sin(self.w * t)])

    def _vel_raw(self, t):
        return np.array([-self.R * self.w * np.sin(self.w * t),
                         self.R * self.w * np.cos(self.w * t)])


# ============================================================
# 直线轨迹
# ============================================================
class LineTrajectory(BaseTrajectory):

    def __init__(self, speed: float, start=(0.0, 0.0), yaw: float = 0.0,
                 T_vis: float = 30.0):
        self.v = speed
        self.p0 = np.array(start, dtype=float)
        self.c, self.s = np.cos(yaw), np.sin(yaw)
        self.T_vis = T_vis
        self.period = None

    def _pose_raw(self, t):
        d = self.v * t
        return self.p0 + np.array([self.c * d, self.s * d])

    def _vel_raw(self, t):
        return np.array([self.v * self.c, self.v * self.s])


# ============================================================
# 8 字轨迹（Gerono 双纽线）
# ============================================================
class LemniscateTrajectory(BaseTrajectory):

    def __init__(self, a: float, omega: float):
        self.a = a
        self.w = omega
        self.period = 2.0 * np.pi / omega

    def _pose_raw(self, t):
        return np.array([self.a * np.cos(self.w * t),
                         0.5 * self.a * np.sin(2.0 * self.w * t)])

    def _vel_raw(self, t):
        return np.array([-self.a * self.w * np.sin(self.w * t),
                         self.a * self.w * np.cos(2.0 * self.w * t)])


# ============================================================
# 正弦轨迹
# ============================================================
class SineTrajectory(BaseTrajectory):

    def __init__(self, speed: float, amplitude: float, wavelength: float,
                 start=(0.0, 0.0), T_vis: float = 30.0):
        self.v = speed
        self.A = amplitude
        self.k = 2.0 * np.pi / wavelength
        self.p0 = np.array(start, dtype=float)
        self.T_vis = T_vis
        self.period = None

    def _pose_raw(self, t):
        x = self.v * t
        return self.p0 + np.array([x, self.A * np.sin(self.k * x)])

    def _vel_raw(self, t):
        x = self.v * t
        return np.array([self.v,
                         self.A * self.k * self.v * np.cos(self.k * x)])


# ============================================================
# 工厂函数：由 YAML 配置创建轨迹
# ============================================================
_TRAJECTORY_REGISTRY = {
    "circle":     CircleTrajectory,
    "line":       LineTrajectory,
    "lemniscate": LemniscateTrajectory,
    "sine":       SineTrajectory,
}


def make_trajectory(cfg: dict) -> BaseTrajectory:
    """
    cfg 为 YAML 中 reference 段：

    """
    traj_type = cfg.get("type", "circle")
    if traj_type not in _TRAJECTORY_REGISTRY:
        raise ValueError(
            f"未知轨迹类型 '{traj_type}'，可选: {list(_TRAJECTORY_REGISTRY)}")
    params = cfg.get(traj_type, {})
    return _TRAJECTORY_REGISTRY[traj_type](**params)
