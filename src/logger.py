"""仿真数据日志容器"""
import numpy as np


class SimLogger:
    """预分配数组的日志容器，避免逐步 append 的开销"""

    def __init__(self, n_steps: int, dt: float):
        n = n_steps
        self.dt = dt
        self.q = np.zeros((3, n))
        self.qr = np.zeros((3, n))
        self.qe = np.zeros((3, n))
        self.phi_cmd = np.zeros((2, n))      # ★ 下发的轮速指令 [rad/s]（速度控制语义）
        self.vref = np.zeros((2, n))         # 避障融合后的最终指令
        self.vref_track = np.zeros((2, n))   # CascadeFTC 原始跟踪指令
        self.vest = np.zeros((2, n))
        self.thff = np.zeros(n)
        self.thfb = np.zeros(n)
        self.alpha = np.zeros(n)
        self.atil = np.zeros(n)
        self.aff = np.zeros(n)
        self.b = np.zeros(n)
        self.beta = np.zeros(n)
        self.rho = np.zeros(n)
        self.vd = np.zeros(n)
        self.d_obs = np.zeros(n)             # 最近障碍间隙距离 [m]
        self.priority = np.zeros(n)          # 任务优先级 0=跟踪 1=避障
        self.k = 0

    def log(self, q, qr, qe, phi_cmd, vref, dbg,
            vref_track=None, d_obs=np.inf, priority=0):
        k = self.k
        self.q[:, k] = q
        self.qr[:, k] = qr
        self.qe[:, k] = qe
        self.phi_cmd[:, k] = phi_cmd
        self.vref[:, k] = vref
        self.vref_track[:, k] = vref_track if vref_track is not None else vref
        self.vest[:, k] = [dbg["vrx_est"], dbg["vry_est"]]
        self.thff[k] = dbg["theta_ff"]
        self.thfb[k] = dbg["theta_fb"]
        self.alpha[k] = dbg["alpha"]
        self.atil[k] = dbg["alpha_til"]
        self.aff[k] = dbg["alpha_ff"]
        self.b[k] = dbg["b"]
        self.beta[k] = dbg["beta"]
        self.rho[k] = dbg["rho"]
        self.vd[k] = dbg["v_d"]
        self.d_obs[k] = d_obs
        self.priority[k] = priority
        self.k += 1

    def trim(self):
        """截断到实际写入长度"""
        k = self.k
        for name in ["q", "qr", "qe", "phi_cmd", "vref", "vref_track", "vest"]:
            setattr(self, name, getattr(self, name)[:, :k])
        for name in ["thff", "thfb", "alpha", "atil", "aff",
                     "b", "beta", "rho", "vd", "d_obs", "priority"]:
            setattr(self, name, getattr(self, name)[:k])
        return self

    @property
    def t_axis(self):
        return np.arange(self.k) * self.dt
