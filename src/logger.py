"""仿真数据日志容器"""
import numpy as np


class SimLogger:
    """预分配数组的日志容器，避免逐步 append 的开销"""

    def __init__(self, n_steps: int, dt: float):
        n = n_steps
        self.dt = dt
        self.q       = np.zeros((3, n))
        self.qr      = np.zeros((3, n))
        self.qe      = np.zeros((3, n))
        self.tau_cmd = np.zeros((2, n))   # 最终下发力矩（含补偿与限幅）
        self.tau_pid = np.zeros((2, n))   # PID 原始输出（不含补偿）
        self.vref    = np.zeros((2, n))
        self.vest    = np.zeros((2, n))
        self.thff    = np.zeros(n)
        self.thfb    = np.zeros(n)
        self.alpha   = np.zeros(n)
        self.atil    = np.zeros(n)
        self.aff     = np.zeros(n)
        self.b       = np.zeros(n)
        self.beta    = np.zeros(n)
        self.rho     = np.zeros(n)
        self.vd      = np.zeros(n)
        self.intw    = np.zeros((2, n))
        self.k = 0

    def log(self, q, qr, qe, tau_cmd, tau_pid, vref, dbg, int_w):
        k = self.k
        self.q[:, k]   = q
        self.qr[:, k]  = qr
        self.qe[:, k]  = qe
        self.tau_cmd[:, k] = tau_cmd
        self.tau_pid[:, k] = tau_pid
        self.vref[:, k] = vref
        self.vest[:, k] = [dbg["vrx_est"], dbg["vry_est"]]
        self.thff[k]   = dbg["theta_ff"]
        self.thfb[k]   = dbg["theta_fb"]
        self.alpha[k]  = dbg["alpha"]
        self.atil[k]   = dbg["alpha_til"]
        self.aff[k]    = dbg["alpha_ff"]
        self.b[k]      = dbg["b"]
        self.beta[k]   = dbg["beta"]
        self.rho[k]    = dbg["rho"]
        self.vd[k]     = dbg["v_d"]
        self.intw[:, k] = int_w
        self.k += 1

    def trim(self):
        """截断到实际写入长度"""
        k = self.k
        for name in ["q", "qr", "qe", "tau_cmd", "tau_pid", "vref", "vest", "intw"]:
            setattr(self, name, getattr(self, name)[:, :k])
        for name in ["thff", "thfb", "alpha", "atil",
                     "aff", "b", "beta", "rho", "vd"]:
            setattr(self, name, getattr(self, name)[:k])
        return self

    @property
    def t_axis(self):
        return np.arange(self.k) * self.dt
