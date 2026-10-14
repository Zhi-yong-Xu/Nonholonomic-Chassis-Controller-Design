import time
from pathlib import Path

import numpy as np
import yaml
import mujoco
import mujoco.viewer

from src import (FTDSO, CascadeFTC, SimLogger, HuskySim, make_trajectory,
                 add_line, add_sphere, MuJoCoObstacleSensor,
                 NullSpaceObstacleTask, TaskPriorityCoordinator)

# ============================================================
# 项目根目录定位（保证任意目录启动均正确）
# ============================================================
ROOT = Path(__file__).resolve().parent


def load_config(path: str = None) -> dict:
    path = path or (ROOT / "config" / "husky.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 将相对路径解析为绝对路径
    xml = Path(cfg["model"]["xml_path"])
    if not xml.is_absolute():
        cfg["model"]["xml_path"] = str((ROOT / xml).resolve())
    return cfg


# ============================================================
# 主控制循环
# ============================================================
def run_sim(cfg: dict):
    dt = cfg["model"]["timestep"]
    n_steps = int(cfg["sim"]["duration"] / dt)

    # --- 仿真环境（内含 MJCF 参数提取与校验）---
    sim = HuskySim(cfg)
    print("[MJCF 提取参数]")
    for k_, v_ in sim.params.items():
        print(f" {k_:>14s} = {v_:.6f}")
    print(f" Omega 矩阵:\n{sim.Omega}")

    # --- 参考轨迹 ---
    traj = make_trajectory(cfg["reference"])
    print(f"[参考轨迹] type={cfg['reference']['type']}, period={traj.period}")

    # --- 控制器 / 观测器 ---
    ftdso_cfg = dict(cfg["ftdso"])
    ftdso_cfg["init"] = traj.initial_pos()
    ftdso = FTDSO(ftdso_cfg, dt)
    ctrl = CascadeFTC(cfg["position_loop"], cfg["attitude_loop"],
                      cfg["command_limits"], ftdso, dt)

    # --- 障碍感知（几何解析：d 为扣除车体半径 r_c 后的间隙距离）---
    obs_sensor = MuJoCoObstacleSensor(
        sim.model, sim.data,
        robot_body=cfg["avoidance"].get("robot_body", "base_link"),
        distmax=cfg["avoidance"]["distmax"],
        robot_radius=cfg["avoidance"].get("robot_radius", 0.6),
    )

    # --- 零空间避障协调器 ---
    obs_task = NullSpaceObstacleTask(cfg["avoidance"])
    coordinator = TaskPriorityCoordinator(
        cfg["avoidance"], obs_task,
        v_max=cfg["command_limits"]["v_max"],
        w_max=cfg["command_limits"]["w_max"])

    # --- 日志 ---
    logger = SimLogger(n_steps, dt)

    # --- 可视化轨迹点 ---
    traj_vis = traj.vis_points(n=241, z=0.02)

    wall_t0 = time.time()
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        k = 0
        try:
            while viewer.is_running() and k < n_steps:
                t = k * dt

                # ---- 状态读取 ----
                q = sim.get_pose()

                # ---- 参考轨迹 ----
                qr = traj.pose(t)

                # ---- 级联控制器（轨迹跟踪，内含 FTDSO 更新）----
                Vref_track, qe, dbg = ctrl.compute(qr, q)

                # ---- 障碍感知 + 零空间融合 ----
                # ★ nearest_point 返回 (p_o, d, c_o) 三元组
                p_o, d, c_o = obs_sensor.nearest_point()
                # ★ 零空间合成：s=0 直通；s=1 时 q̇_cmd = q̇_o + N_o·R(θ)·v_m
                # ★ 传入 qr 供避障向量判决（参考点 q_t = qr 的 x,y）
                Vref = coordinator.compute(q, qr, Vref_track, p_o, d, c_o)

                # ---- 运动学逆映射 + 轮速指令下发（速度控制语义）----
                Phi_cmd = sim.Omega @ Vref          # 轮速指令 [右, 左] [rad/s]
                sim.apply_wheel_velocities(Phi_cmd[0], Phi_cmd[1])

                # ---- 日志 ----
                logger.log(q, qr, qe, Phi_cmd, Vref, dbg,
                           vref_track=Vref_track,
                           d_obs=d, priority=coordinator.priority)

                # ---- 可视化 ----
                scn = viewer.user_scn
                scn.ngeom = 0
                for i in range(len(traj_vis) - 1):
                    add_line(scn, traj_vis[i], traj_vis[i + 1], 0.005,
                             np.array([1.0, 0.0, 0.0, 0.6]))
                add_sphere(scn, np.array([qr[0], qr[1], 0.15]), 0.04,
                           np.array([1.0, 1.0, 0.0, 1.0]))
                add_sphere(scn, np.array([q[0], q[1], 0.15]), 0.04,
                           np.array([0.0, 1.0, 0.0, 1.0]))
                add_line(scn, np.array([qr[0], qr[1], 0.15]),
                         np.array([q[0], q[1], 0.15]), 0.008,
                         np.array([1.0, 1.0, 1.0, 0.8]))
                # 避障激活时：车体到最近障碍点画一条蓝色连线
                if p_o is not None:
                    add_line(scn, np.array([q[0], q[1], 0.15]),
                             np.array([p_o[0, 0], p_o[1, 0], 0.15]), 0.006,
                             np.array([0.0, 0.4, 1.0, 0.9]))

                # ---- 步进 + 实时同步 ----
                sim.step()
                viewer.sync()
                if cfg["sim"]["realtime"]:
                    elapsed = time.time() - wall_t0
                    if t > elapsed:
                        time.sleep(t - elapsed)
                k += 1
        except Exception as e:
            print(f"[提示] 可视化中断: {e}")

    if k == 0:
        print("[提示] 一步都未执行即退出，无数据可绘图。")
        return None
    print(f"[仿真结束] 实际运行 {k} 步 / {k * dt:.2f} s "
          f"(计划 {n_steps} 步 / {n_steps * dt:.1f} s)")
    return logger.trim()


# ============================================================
# 绘图
# ============================================================
def plot_results(log: SimLogger, cfg: dict, traj):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    traj_name = cfg["reference"]["type"]
    t = log.t_axis

    # ---- 轨迹 ----
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(log.qr[0], log.qr[1], 'b-', lw=1.5, label='参考')
    axs[0, 0].plot(log.q[0], log.q[1], 'r--', lw=1.5, label='实际')
    axs[0, 0].set_xlabel('X [m]'); axs[0, 0].set_ylabel('Y [m]')
    axs[0, 0].set_title(f'轨迹跟踪 ({traj_name})'); axs[0, 0].legend()
    axs[0, 0].grid(True); axs[0, 0].axis('equal')

    axs[0, 1].plot(t, log.qr[0] - log.q[0], 'b', label='e_x')
    axs[0, 1].plot(t, log.qr[1] - log.q[1], 'r', label='e_y')
    axs[0, 1].set_xlabel('Time [s]'); axs[0, 1].set_ylabel('误差 [m]')
    axs[0, 1].set_title('位置误差'); axs[0, 1].legend(); axs[0, 1].grid(True)

    axs[1, 0].plot(t, log.vref_track[0], 'b--', label='v 跟踪指令')
    axs[1, 0].plot(t, log.vref[0], 'b', label='v 融合后')
    axs[1, 0].set_xlabel('Time [s]'); axs[1, 0].set_ylabel('v [m/s]')
    axs[1, 0].set_title('线速度（虚线=跟踪，实线=避障融合后）')
    axs[1, 0].legend(); axs[1, 0].grid(True)

    axs[1, 1].plot(t, log.vref_track[1], 'b--', label='ω 跟踪指令')
    axs[1, 1].plot(t, log.vref[1], 'b', label='ω 融合后')
    axs[1, 1].set_xlabel('Time [s]'); axs[1, 1].set_ylabel('ω [rad/s]')
    axs[1, 1].set_title('角速度指令'); axs[1, 1].legend(); axs[1, 1].grid(True)
    plt.suptitle('MuJoCo: CascadeFTC 轨迹跟踪 + 零空间避障')
    plt.tight_layout()

    # ---- 避障监控 ----
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(t, log.d_obs, 'k', lw=1.5, label='d_obs (间隙距离)')
    axs[0].axhline(cfg["avoidance"]["safe_d"], color='r', ls='--',
                   label='safe_d')
    axs[0].set_ylabel('间隙距离 [m]')
    axs[0].set_title('最近障碍间隙距离与任务优先级')
    axs[0].legend(); axs[0].grid(True)
    axs[1].plot(t, log.priority, 'k', lw=1.5)
    axs[1].set_ylim([-0.2, 1.2]); axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['跟踪 TRACK', '避障 AVOID'])
    axs[1].set_xlabel('Time [s]'); axs[1].grid(True)
    plt.tight_layout()

    # ---- 模式判据 β ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(t, log.beta, 'k', lw=1.5)
    axs[0].axhline(np.pi / 2, color='r', ls='--')
    axs[0].axhline(-np.pi / 2, color='r', ls='--')
    axs[0].set_xlabel('Time [s]'); axs[0].set_ylabel(r'$\beta$ [rad]')
    axs[0].set_title(r'模式判据 $\beta=\mathrm{wrap}(\theta_{fb}-\theta_{ff})$')
    axs[0].grid(True)
    axs[1].plot(t, log.b, 'k', lw=1.5)
    axs[1].set_ylim([-0.2, 1.2]); axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['前向 FWD', '反向 REV'])
    axs[1].set_xlabel('Time [s]'); axs[1].set_title('模式变量 b')
    axs[1].grid(True)
    plt.tight_layout()

    # ---- FTDSO ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(t, log.vd, 'b', lw=1.5)
    axs[0].set_xlabel('Time [s]'); axs[0].set_ylabel(r'$v_d$ [m/s]')
    axs[0].set_title('位置环输出 v_d'); axs[0].grid(True)
    v_true = np.array([traj.velocity(ti) for ti in t]).T
    axs[1].plot(t, v_true[0], 'b', label=r'$v_{rx}$真')
    axs[1].plot(t, log.vest[0], 'r--', label=r'$v_{rx}$估')
    axs[1].plot(t, v_true[1], 'g', label=r'$v_{ry}$真')
    axs[1].plot(t, log.vest[1], 'm--', label=r'$v_{ry}$估')
    axs[1].set_xlabel('Time [s]'); axs[1].set_ylabel('v [m/s]')
    axs[1].set_title('FTDSO 速度估计'); axs[1].legend(); axs[1].grid(True)
    plt.tight_layout()

    # ---- 轮速指令（速度控制语义）----
    plt.figure(figsize=(9, 4))
    plt.plot(t, log.phi_cmd[0], 'b', label='右轮 速度指令')
    plt.plot(t, log.phi_cmd[1], 'r', label='左轮 速度指令')
    plt.xlabel('Time [s]'); plt.ylabel(r'$\Phi_{cmd}$ [rad/s]')
    plt.title('轮速指令（Ω·Vref 逆映射直通下发）')
    plt.legend(); plt.grid(True)
    plt.tight_layout()

    plt.show()


# ============================================================
# 入口（全文件仅此一处）
# ============================================================
if __name__ == "__main__":
    cfg = load_config()
    traj = make_trajectory(cfg["reference"])
    logger = run_sim(cfg)
    if logger is not None:
        plot_results(logger, cfg, traj)
