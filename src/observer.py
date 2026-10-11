"""有限时间干扰观测器 (FTDSO)"""
import numpy as np

from .utils import sig


class FTDSO:


    def __init__(self, cfg: dict, dt: float):
        self.K_o1 = cfg["K_o1"]
        self.K_o2 = cfg["K_o2"]
        self.p_o1 = cfg["p_o1"]
        self.p_o2 = cfg["p_o2"]
        self.pos_hat = np.array(cfg["init"], dtype=float)  # [xr_hat, yr_hat]
        self.dt = dt

    def update(self, qr: np.ndarray) -> np.ndarray:
        """
        输入参考点真实位置 qr = [xr, yr]
        返回估计速度 [vrx_est, vry_est]，并内部更新估计位置。
        """
        e_o = qr[:2] - self.pos_hat
        v_est = self.K_o1 * sig(e_o, self.p_o1) + self.K_o2 * sig(e_o, self.p_o2)

        self.pos_hat += v_est * self.dt
        return v_est

    @property
    def estimated_pos(self) -> np.ndarray:
        return self.pos_hat.copy()

    def reset(self):
        self.pos_hat = np.zeros(2)
