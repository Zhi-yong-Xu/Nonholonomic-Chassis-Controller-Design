"""级联轨迹跟踪控制器 + 内环轮速 PID"""
import numpy as np

from .utils import sig, wrap_to_pi


class CascadeFTC:
    """
    级联 FTC 控制器：
      - FTDSO 前馈估计
      - 双幂次位置环
      - 前馈 + 双幂次姿态环
      - FWD/REV 模式切换
    输出：车体系速度指令 Vref = [v_c, omega_c]
    """

    def __init__(self, pos_cfg: dict, att_cfg: dict, lim_cfg: dict,
                 ftdso, dt: float):
        # 位置环
        self.K_R1 = pos_cfg["K_R1"]; self.K_R2 = pos_cfg["K_R2"]
        self.P1   = pos_cfg["P1"];   self.P2   = pos_cfg["P2"]
        # 姿态环
        self.K_A0 = att_cfg["K_A0"]
        self.K_A1 = att_cfg["K_A1"]; self.K_A2 = att_cfg["K_A2"]
        # 限幅
        self.v_max = lim_cfg["v_max"]
        self.w_max = lim_cfg["w_max"]
        # 观测器（注入）
        self.observer = ftdso
        self.dt = dt

    def compute(self, qr: np.ndarray, q: np.ndarray):
        """
        qr: 参考位姿 [xr, yr, thr]
        q : 实际位姿 [x, y, yaw]
        返回 (Vref, qe, dbg)
        """
        ex, ey = qr[0] - q[0], qr[1] - q[1]
        rho = np.hypot(ex, ey)

        # ---- FTDSO ----
        vrx_est, vry_est = self.observer.update(qr)

        # ---- 期望航向 ----
        theta_fb = np.arctan2(ey, ex)                       # 反馈航向
        theta_ff = np.arctan2(vry_est, vrx_est)             # 前馈航向

        # ---- 航向误差 ----
        alpha    = wrap_to_pi(theta_fb - q[2])
        alpha_ff = wrap_to_pi(theta_ff - q[2])

        # ---- 模式切换判据（无滞回）----
        beta = wrap_to_pi(theta_fb - theta_ff)
        b = 1 if np.abs(beta) > np.pi / 2 else 0
        alpha_til = wrap_to_pi(alpha - b * np.pi)

        # ---- 位置环 ----
        v_d =  (self.K_R1 * sig(rho, self.P1) + self.K_R2 * sig(rho, self.P2))
        v_c = v_d * np.cos(alpha)

        # ---- 姿态环（前馈 + 反馈）----
        omega_c = (self.K_A0 * alpha_ff / (rho+1e-6)
                   + self.K_A1 * sig(rho * alpha_til, self.P1)
                   + self.K_A2 * sig(rho * alpha_til, self.P2))

        # ---- 指令限幅 ----
        v_c     = np.clip(v_c, -self.v_max, self.v_max)
        omega_c = np.clip(omega_c, -self.w_max, self.w_max)

        Vref = np.array([v_c, omega_c])

        # ---- 车体系误差（用于记录/分析）----
        eth = wrap_to_pi(qr[2] - q[2])
        ct, st = np.cos(q[2]), np.sin(q[2])
        R = np.array([[ct, st, 0.0], [-st, ct, 0.0], [0.0, 0.0, 1.0]])
        qe = R @ np.array([ex, ey, eth])

        dbg = dict(alpha=alpha, alpha_til=alpha_til, alpha_ff=alpha_ff,
                   theta_ff=theta_ff, theta_fb=theta_fb, b=b, beta=beta,
                   vrx_est=vrx_est, vry_est=vry_est,
                   rho=rho, v_d=v_d)

        return Vref, qe, dbg


