import numpy as np
from .utils import wrap_to_pi


class NullSpaceObstacleTask:
    FREE = 0    # 锁存自由态（未进入影响区）

    def __init__(self, cfg: dict):
        self.k_p = cfg["k_p"]                      # 势场增益（径向与切向共用）
        self.kappa = float(cfg.get("kappa", 1.0))  # 切向/径向速率比 κ（无量纲）
        self.k_theta = cfg["k_theta"]              # 航向通道增益 [1/s]
        self.safe_d = cfg["safe_d"]                # 影响区阈值 d_safe [m]（唯一判断条件）
        self.r_c = cfg.get("robot_radius", 0.6)    # 车体等效圆盘半径 [m]
        self.d_min = cfg.get("d_min", 0.05)        # d 数值下限 [m]，防 1/d³ 发散
        # =================== 开关：侧向判决准则 ===================
        judge = str(cfg.get("side_judge", "target")).lower()
        if judge not in ("target", "rotation"):
            raise ValueError(
                f"side_judge 取值必须是 'target' 或 'rotation'，收到: {judge!r}")
        self.side_judge = judge
        # ==========================================================
        self.sigma_L = self.FREE                   # 锁存变量 σ_L ∈ {+1,-1,FREE}

    # ---- 侧向判决：准则由 self.side_judge 选择（判决函数，不含状态）----
    def choose_side(self, p: np.ndarray, c_o: np.ndarray,
                    q_t: np.ndarray = None, theta: float = None) -> int:
        # 视线单位向量 ĝ 与左切向 u⁺ = R(π/2)·ĝ（两准则共用）
        g = (c_o - p) / max(np.linalg.norm(c_o - p), 1e-9)
        u_plus = np.array([[-g[1, 0]], [g[0, 0]]])

        if self.side_judge == "target":
            # 目标方向最小夹角：内积符号判决（与当前航向 θ 无关）
            e_t = q_t - p                                     # 机器人 → 参考点
            proj = float((u_plus.T @ e_t)[0, 0])              # ‖e_t‖·cos δ
            return 1 if proj >= 0 else -1
        else:
            # 最小转角：基于当前航向 θ 与视线方位角 φ
            phi = float(np.arctan2(float(c_o[1]) - float(p[1]),
                                   float(c_o[0]) - float(p[0])))
            return 1 if np.sin(theta - phi) >= 0.0 else -1

    # ---- 锁存状态机：单一判断条件的 σ_L 转移（每个控制周期调用一次）----
    def update_latch(self, q: np.ndarray, p_o, c_o, qr: np.ndarray, d: float):
        d_eff = np.inf if p_o is None else float(d)

        if self.sigma_L == self.FREE:
            if d_eff <= self.safe_d:
                # 进入事件：按配置准则判决一次，锁存
                p = np.array([q[0], q[1]]).reshape(2, 1)
                self.sigma_L = self.choose_side(
                    p, c_o, q_t=qr[:2].reshape(2, 1), theta=float(q[2]))
        else:
            if d_eff >= self.safe_d:
                # 退出事件：单一判断条件释放
                self.sigma_L = self.FREE

    # ---- 径向势场速率（无限幅）----
    def _g_r(self, d: float) -> float:
        """g_r(d) = k_p / d³ ≥ 0，原样输出，不做饱和。"""
        return float(self.k_p / max(d, self.d_min) ** 3)

    # ---- 切向逃逸速率（与径向同增益同律，比例 κ，无限幅）----
    def _g_tau(self, d: float) -> float:
        """g_τ(d) = κ·k_p / d³ ≥ 0，原样输出，不做饱和。κ=0 → 纯径向。"""
        return float(self.kappa * self.k_p / max(d, self.d_min) ** 3)

    def compute(self, q: np.ndarray, p_o: np.ndarray, c_o: np.ndarray,
                qr: np.ndarray, d: float):

        p = np.array([q[0], q[1]]).reshape(2, 1)
        theta = float(q[2])
        d = max(float(d), 1e-3)

        # ---- 雅可比：第一行 = ∇d = ûᵀ（背离障碍的梯度），第二行 = [0,0,1] ----
        u = ((p - p_o) / max(np.linalg.norm(p - p_o), 1e-6)).reshape(1, 2)
        J_o = np.vstack((
            np.hstack((u, np.zeros((1, 1)))),
            np.array([[0.0, 0.0, 1.0]]),
        ))

        # ---- 逃逸航向：使用锁存的 σ_L 构造 θ_d（绕行中不重判）----
        phi = float(np.arctan2(float(c_o[1]) - float(q[1]),
                               float(c_o[0]) - float(q[0])))
        theta_d = phi + self.sigma_L * 0.5 * np.pi

        # ---- 平面逃逸速度：g_r·û + g_τ·t̂（均无限幅）----
        g_r = self._g_r(d)
        g_t = self._g_tau(d)
        t_hat = np.array([[np.cos(theta_d)],
                          [np.sin(theta_d)]])                     # 切向逃逸方向 t̂
        v_xy = g_r * u.reshape(2, 1) + g_t * t_hat                # (2,1)

        e_th = wrap_to_pi(theta_d - theta)
        v_dot_o = np.array([
            [float(v_xy[0, 0])],
            [float(v_xy[1, 0])],
            [self.k_theta * e_th],
        ])

        # ---- 形状加固：输出恒为 (3,1) 列向量 ----
        v_dot_o = np.asarray(v_dot_o, dtype=float).reshape(3, 1)
        return v_dot_o, J_o, d, self.sigma_L, theta_d


class TaskPriorityCoordinator:
    """
      d ≤ d_safe → s=1,避障升为主任务：
      d > d_safe → s=0,跟踪全权(CascadeFTC 输出直通)，避障精确归零
    """

    def __init__(self, cfg: dict, obs_task: NullSpaceObstacleTask,
                 v_max: float, w_max: float):
        self.task = obs_task
        self.v_max = float(v_max)   # 最终线速度限幅
        self.w_max = float(w_max)   # 最终角速度限幅
        self.priority = 0                 # s ∈ {0, 1}
        self.sigma = 0                    # 最近一次锁存状态 σ_L（供调试/记录）

    def compute(self, q: np.ndarray, qr: np.ndarray, Vref_track: np.ndarray,
                p_o, d: float, c_o) -> np.ndarray:

        th = float(q[2])
        c, s = np.cos(th), np.sin(th)
        # ---- 输出映射 R(θ)：车体系 (v, ω) → 构型空间速度 ----
        R = np.array([[c, 0.0],
                      [s, 0.0],
                      [0.0, 1.0]])

        # ---- ① 跟踪任务速度：q̇_m = R(θ)·v_m ----
        V_track = np.asarray(Vref_track, dtype=float).reshape(2, 1)
        v_dot_m = R @ V_track                             # (3,1)

        # ---- ② 锁存状态机维护（每周期，无论 s 取值）----
        self.task.update_latch(q, p_o, c_o, qr, d)
        self.sigma = self.task.sigma_L

        # ---- ③ 任务切换：纯间隙距离单阈值 ----
        if p_o is not None and d <= self.task.safe_d:
            self.priority = 1
        else:
            self.priority = 0

        # ---- 零空间链式合成 ----
        if self.priority == 1 and p_o is not None:
            # s=1：q̇_cmd = 1[σ_L≠FREE]·q̇_o + N_o·R(θ)·v_m
            v_dot_o, J_o, _, _, _ = self.task.compute(q, p_o, c_o, qr, d)
            N_o = np.eye(3) - J_o.T @ J_o                
            if self.task.sigma_L != self.task.FREE:
                v_des = v_dot_o + N_o @ v_dot_m
            else:
                v_des = N_o @ v_dot_m
        else:
            # s=0：跟踪全权，避障精确归零
            v_des = v_dot_m

        v_des = np.asarray(v_des, dtype=float).reshape(3, 1)   # 兜底加固

        # ---- 回投 (v, ω) + 最终指令限幅 ----
        Vref_safe = (R.T @ v_des).flatten()               # (2,)
        Vref_safe = np.clip(Vref_safe,
                            [-self.v_max, -self.w_max],
                            [+self.v_max, +self.w_max])
        return Vref_safe
