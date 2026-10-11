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
        v_d = self.K_R1 * sig(rho, self.P1) + self.K_R2 * sig(rho, self.P2)
        v_c = v_d * np.cos(alpha)

        # ---- 姿态环（前馈 + 反馈）----
        omega_c = (self.K_A0 * alpha_ff
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


class WheelSpeedPID:
    """
    轮速 PID + 死区补偿 + 力矩限幅，双轮向量化实现。

    完整执行链：
        Phi_ref --> PID(P+I+滤波D) --> 死区补偿(按 ω 符号) --> 力矩限幅 --> tau_cmd
    """

    def __init__(self, cfg: dict, dt: float):
        # PID 增益
        self.kp = cfg["kp"]
        self.ki = cfg["ki"]
        self.kd = cfg["kd"]
        self.int_max = cfg["int_max"]
        self.d_filt_alpha = cfg["d_filt_alpha"]
        # 执行器补偿参数
        self.tau_max = cfg["tau_max"]
        self.tau_dz  = cfg["tau_dz"]
        self.w_deadband = cfg.get("w_deadband", 1e-3)   # ω 符号判断死区阈值
        self.dt = dt

        # 内部状态
        self.dPhi_filt = np.zeros(2)
        self.Phi_prev  = np.zeros(2)
        self.int_w     = np.zeros(2)
        # 最近一次的输出（供日志/调试）
        self.last_tau_pid  = np.zeros(2)
        self.last_tau_cmd  = np.zeros(2)

    # ---- PID 本体 ----
    def _pid(self, Phi_ref: np.ndarray, Phi: np.ndarray) -> np.ndarray:
        err = Phi_ref - Phi

        # 积分项（带限幅）
        self.int_w = np.clip(self.int_w + err * self.dt,
                             -self.int_max, self.int_max)

        # 微分项（一阶低通滤波）
        dPhi_raw = (Phi - self.Phi_prev) / self.dt
        self.dPhi_filt = (self.d_filt_alpha * self.dPhi_filt
                          + (1.0 - self.d_filt_alpha) * dPhi_raw)

        tau = self.kp * err + self.ki * self.int_w - self.kd * self.dPhi_filt

        self.Phi_prev = Phi.copy()
        return tau

    # ---- 死区补偿 + 左右轮力矩分配 ----
    def _deadzone_compensation(self, tau: np.ndarray,
                               w_c: float) -> np.ndarray:
        """
        按角速度指令 w_c 的符号施加死区补偿力矩：
          右轮 += 0.5 * tau_dz * sgn(w_c)
          左轮 -= 0.5 * tau_dz * sgn(w_c)
        最后做总力矩限幅。
        """
        if w_c > self.w_deadband:
            sgn = 1.0
        elif w_c < -self.w_deadband:
            sgn = -1.0
        else:
            sgn = 0.0
        tau_comp = self.tau_dz * sgn

        tau_R = np.clip(tau[0] + 0.5 * tau_comp, -self.tau_max, self.tau_max)
        tau_L = np.clip(tau[1] - 0.5 * tau_comp, -self.tau_max, self.tau_max)
        return np.array([tau_R, tau_L])

    # ---- 完整对外接口：PID + 补偿 + 限幅 ----
    def compute_command(self, Phi_ref: np.ndarray, Phi: np.ndarray,
                        w_c: float) -> np.ndarray:
        """
        输入:
            Phi_ref : 期望轮速 [右, 左] [rad/s]
            Phi     : 实际轮速 [右, 左] [rad/s]
            w_c     : 角速度指令 [rad/s]（用于死区补偿符号判断）
        返回:
            tau_cmd = [tau_R, tau_L]，最终下发电动力矩 [N·m]
        """
        tau = self._pid(Phi_ref, Phi)
        self.last_tau_pid = tau
        tau_cmd = self._deadzone_compensation(tau, w_c)
        self.last_tau_cmd = tau_cmd
        return tau_cmd

    def reset(self):
        self.dPhi_filt = np.zeros(2)
        self.Phi_prev  = np.zeros(2)
        self.int_w     = np.zeros(2)
        self.last_tau_pid = np.zeros(2)
        self.last_tau_cmd = np.zeros(2)
