
import time
from pathlib import Path

import numpy as np
import yaml
import mujoco
import mujoco.viewer

from src import (FTDSO, CascadeFTC, WheelSpeedPID, SimLogger, HuskySim,
                 make_trajectory, add_line, add_sphere)


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
    for k_, v in sim.params.items():
        print(f"  {k_:>14s} = {v:.6f}")
    print(f"  Omega 矩阵:\n{sim.Omega}")

    # --- 参考轨迹（工厂函数创建）---
    traj = make_trajectory(cfg["reference"])
    print(f"[参考轨迹] type={cfg['reference']['type']}, "
          f"period={traj.period}")

    # --- 控制器 / 观测器 ---
    # FTDSO 初始估计位置自动对齐轨迹起点
    ftdso_cfg = dict(cfg["ftdso"])
    ftdso_cfg["init"] = traj.initial_pos()
    ftdso = FTDSO(ftdso_cfg, dt)

    ctrl = CascadeFTC(cfg["position_loop"], cfg["attitude_loop"],
                      cfg["command_limits"], ftdso, dt)
    pid = WheelSpeedPID(cfg["wheel_pid"], dt)   # 内含死区补偿与力矩限幅

    # --- 日志 ---
    logger = SimLogger(n_steps, dt)

    # --- 可视化轨迹点：由轨迹类自己生成 ---
    traj_vis = traj.vis_points(n=241, z=0.02)

    wall_t0 = time.time()

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        k = 0
        try:
            # while 循环：窗口关闭（is_running=False）或跑满 n_steps 时退出
            while viewer.is_running() and k < n_steps:
                t = k * dt

                # ---- 状态读取 ----
                q   = sim.get_pose()
                Phi = sim.get_wheel_speeds()

                # ---- 参考轨迹 ----
                qr = traj.pose(t)

                # ---- 级联控制器（内含 FTDSO 更新）----
                Vref, qe, dbg = ctrl.compute(qr, q)

                # ---- 轮速映射 + PID + 死区补偿 + 力矩限幅 ----
                Phi_ref = sim.Omega @ Vref
                tau_cmd = pid.compute_command(Phi_ref, Phi, w_c=Vref[1])

                # ---- 力矩下发 ----
                sim.apply_wheel_torques(tau_cmd[0], tau_cmd[1])

                # ---- 日志 ----
                logger.log(q, qr, qe, tau_cmd, pid.last_tau_pid,
                           Vref, dbg, pid.int_w)

                # ---- 可视化 ----
                scn = viewer.user_scn
                scn.ngeom = 0
                for i in range(len(traj_vis) - 1):
                    add_line(scn, traj_vis[i], traj_vis[i + 1],
                             0.005, np.array([1.0, 0.0, 0.0, 0.6]))
                add_sphere(scn, np.array([qr[0], qr[1], 0.15]),
                           0.04, np.array([1.0, 1.0, 0.0, 1.0]))
                add_sphere(scn, np.array([q[0], q[1], 0.15]),
                           0.04, np.array([0.0, 1.0, 0.0, 1.0]))
                add_line(scn,
                         np.array([qr[0], qr[1], 0.15]),
                         np.array([q[0],   q[1],   0.15]),
                         0.008, np.array([1.0, 1.0, 1.0, 0.8]))

                # ---- 步进 + 实时同步 ----
                sim.step()
                viewer.sync()

                if cfg["sim"]["realtime"]:
                    elapsed = time.time() - wall_t0
                    if t > elapsed:
                        time.sleep(t - elapsed)

                k += 1

        except Exception as e:
            # 捕获窗口关闭瞬间可能的渲染/同步异常，保证干净退出
            print(f"[提示] 可视化中断: {e}")

    # ---- 结束信息 ----
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
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Microsoft YaHei'
    plt.rcParams['mathtext.it'] = 'Microsoft YaHei:italic'
    plt.rcParams['mathtext.bf'] = 'Microsoft YaHei:bold'

    t = log.t_axis
    tau_max = cfg["wheel_pid"]["tau_max"]
    int_max = cfg["wheel_pid"]["int_max"]
    traj_name = cfg["reference"]["type"]

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

    axs[1, 0].plot(t, log.vref[0], 'b', label='指令')
    axs[1, 0].set_xlabel('Time [s]'); axs[1, 0].set_ylabel('v [m/s]')
    axs[1, 0].set_title('线速度（负值=倒车）'); axs[1, 0].legend(); axs[1, 0].grid(True)

    axs[1, 1].plot(t, log.vref[1], 'b', label='ω 指令')
    axs[1, 1].set_xlabel('Time [s]'); axs[1, 1].set_ylabel('ω [rad/s]')
    axs[1, 1].set_title('角速度指令'); axs[1, 1].legend(); axs[1, 1].grid(True)
    plt.suptitle('MuJoCo: FTDSO + 理论版姿态环 + 内环积分')
    plt.tight_layout()

    # ---- 航向与模式 ----
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].plot(t, log.thfb, 'b', label=r'$\theta_{fb}$')
    axs[0, 0].plot(t, log.thff, 'r--', label=r'$\theta_{ff}$')
    axs[0, 0].plot(t, log.q[2], 'g:', label=r'$\psi$')
    axs[0, 0].set_xlabel('Time [s]'); axs[0, 0].set_ylabel(r'$\theta$ [rad]')
    axs[0, 0].set_title(r'$\theta_{fb},\theta_{ff},\psi$')
    axs[0, 0].legend(); axs[0, 0].grid(True)

    axs[0, 1].plot(t, log.alpha, 'b', label=r'$\alpha$')
    axs[0, 1].plot(t, log.atil, 'r--', label=r'$\alpha_{til}$')
    axs[0, 1].plot(t, log.aff, 'g:', label=r'$\alpha_{ff}$')
    axs[0, 1].set_xlabel('Time [s]'); axs[0, 1].set_ylabel('rad')
    axs[0, 1].set_title(r'$\alpha,\alpha_{til},\alpha_{ff}$')
    axs[0, 1].legend(); axs[0, 1].grid(True)

    axs[1, 0].plot(t, log.rho, 'k')
    axs[1, 0].set_xlabel('Time [s]'); axs[1, 0].set_ylabel(r'$\rho$ [m]')
    axs[1, 0].set_title('距离误差'); axs[1, 0].grid(True)

    axs[1, 1].plot(t, log.b, 'k', lw=1.5)
    axs[1, 1].set_ylim([-0.2, 1.2]); axs[1, 1].set_yticks([0, 1])
    axs[1, 1].set_yticklabels(['前向 FWD', '反向 REV'])
    axs[1, 1].set_xlabel('Time [s]'); axs[1, 1].set_title('模式变量 b')
    axs[1, 1].grid(True)
    plt.suptitle('航向角 / 误差 / 模式')
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

    # ---- v_d 与 FTDSO（真值由轨迹类通用接口给出）----
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

    # ---- 力矩（PID 原始输出 vs 最终下发指令）----
    plt.figure(figsize=(9, 4))
    plt.plot(t, log.tau_pid[0], 'b',  alpha=0.5, label='右轮 PID')
    plt.plot(t, log.tau_cmd[0], 'b-', lw=1.8, label='右轮 下发')
    plt.plot(t, log.tau_pid[1], 'r',  alpha=0.5, label='左轮 PID')
    plt.plot(t, log.tau_cmd[1], 'r-', lw=1.8, label='左轮 下发')
    plt.axhline(tau_max, color='k', ls='--', lw=1)
    plt.axhline(-tau_max, color='k', ls='--', lw=1)
    plt.xlabel('Time [s]'); plt.ylabel(r'$\tau$ [N·m]')
    plt.title('轮力矩：PID原始输出 vs 含死区补偿的下发指令')
    plt.legend(); plt.grid(True)
    plt.tight_layout()

    # ---- 内环积分状态 ----
    plt.figure(figsize=(8, 3))
    plt.plot(t, log.intw[0], 'b', label='右轮积分')
    plt.plot(t, log.intw[1], 'r', label='左轮积分')
    plt.axhline(int_max, color='k', ls='--', lw=1)
    plt.axhline(-int_max, color='k', ls='--', lw=1)
    plt.xlabel('Time [s]'); plt.ylabel(r'$\int e_\phi\,dt$ [rad]')
    plt.title('内环轮速积分状态'); plt.legend(); plt.grid(True)
    plt.tight_layout()

    plt.show()


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    cfg = load_config()
    traj = make_trajectory(cfg["reference"])   # 绘图也需要轨迹对象
    logger = run_sim(cfg)
    if logger is not None:
        plot_results(logger, cfg, traj)
