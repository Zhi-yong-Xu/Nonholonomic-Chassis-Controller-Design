import mujoco
import mujoco.viewer
import numpy as np
import math
import time
import matplotlib.pyplot as plt


# ==========================================
# 辅助函数
# ==========================================
def get_state_from_mjdata(model, data, qpos_adr, dof_adr):
    """返回: 世界系位置(3), 世界系速度(6), 航向角"""
    pos = data.qpos[qpos_adr:qpos_adr + 3].copy()
    vel = data.qvel[dof_adr:dof_adr + 6].copy()

    quat = data.qpos[qpos_adr + 3:qpos_adr + 7]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    R = mat.reshape(3, 3)
    phi = np.arctan2(R[1, 0], R[0, 0])
    return pos, vel, phi


def wrap_angle(a):
    """角度归一化到 [-pi, pi]"""
    return math.atan2(math.sin(a), math.cos(a))


def add_line(scn, p1, p2, radius, rgba):
    """在 viewer 场景中添加一条圆柱线段 (绘制参考轨迹)"""
    if scn.ngeom >= scn.maxgeom:
        return
    mid = (p1 + p2) / 2
    diff = p2 - p1
    length = np.linalg.norm(diff)
    if length < 1e-10:
        return

    direction = diff / length
    z_axis = np.array([0, 0, 1])

    if np.abs(np.dot(direction, z_axis)) > 0.999:
        mat = np.eye(3) if np.dot(direction, z_axis) > 0 else np.diag([1, -1, -1]).astype(float)
    else:
        v = np.cross(z_axis, direction)
        v = v / np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        s = np.sqrt(1 - c * c)
        mat = np.array([
            [c + v[0] * v[0] * (1 - c),        v[0] * v[1] * (1 - c) - v[2] * s,  v[0] * v[2] * (1 - c) + v[1] * s],
            [v[1] * v[0] * (1 - c) + v[2] * s, c + v[1] * v[1] * (1 - c),         v[1] * v[2] * (1 - c) - v[0] * s],
            [v[2] * v[0] * (1 - c) - v[1] * s, v[2] * v[1] * (1 - c) + v[0] * s,  c + v[2] * v[2] * (1 - c)]
        ])

    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, length / 2, 0]),
        mid,
        mat.flatten(),
        rgba
    )
    scn.ngeom += 1


def main():
    # ------------------------------------------
    # 0. 测试模式开关
    #   True : 直线测试 (隔离姿态环, 验证航向律)
    #   False: 圆弧轨迹 (完整验证)
    # ------------------------------------------
    USE_STRAIGHT_TEST = False

    # ------------------------------------------
    # 1. 加载模型与参数提取
    # ------------------------------------------
    model = mujoco.MjModel.from_xml_path(r".\model\husky.xml")
    data = mujoco.MjData(model)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    wheel_names = ['front_left_wheel_link', 'front_right_wheel_link',
                   'rear_left_wheel_link', 'rear_right_wheel_link']
    joint_names = ['front_left_wheel', 'front_right_wheel',
                   'rear_left_wheel', 'rear_right_wheel']

    wheel_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in wheel_names]
    wheel_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    wheel_dof_adrs = [model.jnt_dofadr[j] for j in wheel_joint_ids]

    pos_fl = model.body_pos[wheel_body_ids[0]]
    pos_fr = model.body_pos[wheel_body_ids[1]]
    L_wheel = abs(pos_fl[1] - pos_fr[1])
    d_car = L_wheel / 2.0
    fl_geom_id = model.body_geomadr[wheel_body_ids[0]]
    r = model.geom_size[fl_geom_id][0]

    base_jnt_id = model.body_jntadr[base_id]
    base_qpos_adr = model.jnt_qposadr[base_jnt_id]
    base_dof_adr = model.jnt_dofadr[base_jnt_id]

    # ------------------------------------------
    # 2. 执行器类型自动检测
    # ------------------------------------------
    IS_VELOCITY_SERVO = (model.actuator_biastype[0] == int(mujoco.mjtBias.mjBIAS_AFFINE)
                         and model.actuator_biasprm[0][2] < 0)
    kv_servo = abs(model.actuator_biasprm[0][2]) if IS_VELOCITY_SERVO else 0.0

    ctrl_lo = model.actuator_ctrlrange[:, 0] if model.actuator_ctrllimited[0] else -np.inf
    ctrl_hi = model.actuator_ctrlrange[:, 1] if model.actuator_ctrllimited[0] else np.inf

    print("=" * 60)
    if IS_VELOCITY_SERVO:
        print(f"执行器模式: VELOCITY伺服 (ctrl=目标轮速, kv={kv_servo:.1f})")
    else:
        print("执行器模式: MOTOR/力矩 (ctrl=力矩)")
    print(f"轮半径 r={r:.4f} m, 半轮距 d={d_car:.4f} m")
    print("=" * 60)

    # ------------------------------------------
    # 3. 控制器参数 (全部在世界坐标系下)
    # ------------------------------------------
    # --- 外环位置 PID (世界系误差 -> 世界系制导向量) ---
    # 微分项在世界系下是精确导数: d(ex)/dt = vx_d - vx (无旋转耦合)
    Kp_pos, Ki_pos, Kd_pos = 5.0, 0.1, 1.0
    I_limit = 10.0            # 积分限幅

    # --- 速度指令层 ---
    Kv_fwd = 1.0              # 前向误差投影 -> 线速度增益
    v_max = 2.0               # 线速度上限 (轨迹需求 2 m/s, 留 25% 余量)
    w_max = 5.0               # 角速度上限

    # --- 航向环 ---
    # 期望航向 phi_d = atan2(u_Gy, u_Gx) (世界系), ephi = wrap(phi_d - phi)
    Kp_att = 3.0              # P: 航向偏移
    Kd_att = 0.3              # D: 偏航速率阻尼 (作用于实测 wz)

    # --- motor 模式本地速度环 ---
    kv_pd = 30.0
    tau_max = 100.0

    # --- 世界系积分器状态 ---
    Ix = 0.0
    Iy = 0.0
    phi_d_prev = 0.0          # 期望航向保持值 (防 atan2 小向量抖动)

    # ------------------------------------------
    # 4. 仿真设置与参考轨迹
    # ------------------------------------------
    duration = 60.0
    dt = model.opt.timestep

    radius = 4.0
    omega_traj = 0.25         # 标称线速度 = R * w0 = 2 m/s
    center = np.array([-radius, 0.0])

    # 路径曲率 (前馈 w_ff = v_cmd * kappa)
    kappa = 0.0 if USE_STRAIGHT_TEST else 1.0 / radius

    history = {k: [] for k in
               ['t', 'x', 'y', 'phi', 'phid',
                'err_x', 'err_y', 'err_fwd', 'err_norm', 'ephi',
                'v_cmd', 'w_cmd', 'wz', 'tau_abs_max']}

    # 预生成用于 Viewer 显示的参考轨迹点
    if USE_STRAIGHT_TEST:
        traj_vis_points = [np.array([k, 0.0, 0.2]) for k in np.linspace(0, 30, 61)]
    else:
        traj_vis_points = []
        for th in np.linspace(0, 2 * np.pi, 73):
            traj_vis_points.append(np.array([
                center[0] + radius * np.cos(th),
                center[1] + radius * np.sin(th),
                0.2
            ]))

    # ------------------------------------------
    # 5. 启动可视化与控制循环
    # ------------------------------------------
    mode_str = "Straight-Line Test" if USE_STRAIGHT_TEST else "Circle Trajectory"
    print(f"Launching MuJoCo Viewer (World-Frame Error PID) [{mode_str}]...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        simulation_step = 0

        while viewer.is_running() and data.time < duration:
            t = data.time

            # ==================================================
            # 1. 目标轨迹 (世界系)
            # ==================================================
            if USE_STRAIGHT_TEST:
                x_d = 10.0 * omega_traj * t
                y_d = 0.0
                vx_d = 10.0 * omega_traj
                vy_d = 0.0
            else:
                x_d = center[0] + radius * np.cos(omega_traj * t)
                y_d = center[1] + radius * np.sin(omega_traj * t)
                vx_d = -radius * omega_traj * np.sin(omega_traj * t)
                vy_d = radius * omega_traj * np.cos(omega_traj * t)

            # ==================================================
            # 2. 状态反馈
            # ==================================================
            pos, vel, phi = get_state_from_mjdata(model, data, base_qpos_adr, base_dof_adr)
            x, y = pos[0], pos[1]
            vx = vel[0]
            vy = vel[1]
            wz = vel[5]

            # ==================================================
            # 3. ★ 世界系误差构造 + 外环 PID ★
            #    微分项用 (参考速度 - 实际速度):
            #    世界系下 d(ex)/dt = vx_d - vx 精确成立, 无旋转耦合项
            # ==================================================
            ex = x_d - x
            ey = y_d - y

            # 积分与抗饱和
            Ix = max(min(Ix + ex * dt, I_limit), -I_limit)
            Iy = max(min(Iy + ey * dt, I_limit), -I_limit)

            # 微分 (解析形式, 无差分噪声)
            Dex = vx_d - vx
            Dey = vy_d - vy

            u_Gx = Kp_pos * ex + Ki_pos * Ix + Kd_pos * Dex
            u_Gy = Kp_pos * ey + Ki_pos * Iy + Kd_pos * Dey

            # ==================================================
            # 4. ★ 期望航向: 世界系 PID 向量的方位角 ★
            #    phi_d 是世界系角度, ephi = wrap(phi_d - phi)
            # ==================================================
            if math.hypot(u_Gx, u_Gy) > 1e-3:
                phi_d = math.atan2(u_Gy, u_Gx)
            else:
                phi_d = phi_d_prev     # 制导向量过小(已到目标), 保持上一帧
            phi_d_prev = phi_d

            ephi = wrap_angle(phi_d - phi)

            # ==================================================
            # 5. 前向速度指令
            #    前向误差 = 世界系误差在车头方向的投影 (车体系标量)
            #    大航向偏差时先减速转向
            # ==================================================
            # err_fwd = ex * math.cos(phi) + ey * math.sin(phi)

            # v_cmd = max(min(Kv_fwd * err_fwd, v_max), -v_max)


            c, s = math.cos(phi), math.sin(phi)
            err_fwd = u_Gx * c + u_Gy * s          # ← 用 u_G 而非 ex/ey
            v_cmd = max(min(Kv_fwd * err_fwd, v_max), -v_max)

            v_cmd *= max(math.cos(ephi), 0.0)

            # ==================================================
            # 6. 角速度指令
            #    P(航向偏移) + D(偏航速率阻尼) + 曲率前馈 (v_cmd * kappa)
            # ==================================================
            w_ff = v_cmd * kappa
            w_cmd = Kp_att * ephi - Kd_att * wz + w_ff
            w_cmd = max(min(w_cmd, w_max), -w_max)

            # ==================================================
            # 7. 差速运动学逆解
            # ==================================================
            v_L = v_cmd - w_cmd * d_car
            v_R = v_cmd + w_cmd * d_car
            w_L = v_L / r
            w_R = v_R / r

            # ==================================================
            # 8. 执行器写入 (双模式)
            # ==================================================
            w_wheel_cmds = [w_L, w_R, w_L, w_R]
            tau_abs_max = 0.0
            if IS_VELOCITY_SERVO:
                for i, wc in enumerate(w_wheel_cmds):
                    data.ctrl[i] = max(min(wc, ctrl_hi[i]), ctrl_lo[i])
            else:
                for i, wc in enumerate(w_wheel_cmds):
                    w_meas = data.qvel[wheel_dof_adrs[i]]
                    tau = kv_pd * (wc - w_meas)
                    tau = max(min(tau, tau_max), -tau_max)
                    tau_abs_max = max(tau_abs_max, abs(tau))
                    data.ctrl[i] = tau

            # ==================================================
            # 9. 步进仿真
            # ==================================================
            mujoco.mj_step(model, data)

            # ==================================================
            # 10. 数据记录
            # ==================================================
            history['t'].append(t)
            history['x'].append(x)
            history['y'].append(y)
            history['phi'].append(phi)
            history['phid'].append(phi_d)
            history['err_x'].append(ex)
            history['err_y'].append(ey)
            history['err_fwd'].append(err_fwd)
            history['err_norm'].append(math.hypot(ex, ey))
            history['ephi'].append(ephi)
            history['v_cmd'].append(v_cmd)
            history['w_cmd'].append(w_cmd)
            history['wz'].append(wz)
            history['tau_abs_max'].append(tau_abs_max)

            # ==================================================
            # 11. 绘制参考轨迹和当前目标点
            # ==================================================
            viewer.user_scn.ngeom = 0

            for i in range(len(traj_vis_points) - 1):
                add_line(viewer.user_scn,
                         traj_vis_points[i],
                         traj_vis_points[i + 1],
                         0.02,
                         np.array([1.0, 0.0, 0.0, 0.5]))

            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.1, 0, 0]),
                    np.array([x_d, y_d, 0.2]),
                    np.eye(3).flatten(),
                    np.array([1.0, 1.0, 0.0, 1.0]))
                viewer.user_scn.ngeom += 1

            # ==================================================
            # 12. 同步可视化与实时控制
            # ==================================================
            viewer.sync()

            expected_time = start_time + simulation_step * dt
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(expected_time - current_time)
            simulation_step += 1

    # ------------------------------------------
    # 6. 结果绘图与诊断输出
    # ------------------------------------------
    print("Simulation finished. Plotting results...")

    t_arr = np.array(history['t'])
    ephi_arr = np.array(history['ephi'])
    err_norm_arr = np.array(history['err_norm'])
    tau_arr = np.array(history['tau_abs_max'])

    # 稳态诊断 (取最后 10 s)
    ss_mask = t_arr > (t_arr[-1] - 10.0)
    print("=" * 60)
    print(f"稳态航向偏移 |ephi| (后10s均值): {np.abs(ephi_arr[ss_mask]).mean():.4f} rad "
          f"({np.degrees(np.abs(ephi_arr[ss_mask]).mean()):.2f} deg)")
    print(f"稳态位置误差 norm (后10s均值): {err_norm_arr[ss_mask].mean():.4f} m")
    print(f"峰值力矩: {tau_arr.max():.1f} Nm / 限幅 {tau_max}")
    if tau_arr.max() >= tau_max - 1.0:
        print("⚠ 力矩饱和! 降低轨迹速度或提高 tau_max")
    print("=" * 60)

    fig = plt.figure(figsize=(12, 9))

    # 轨迹跟踪
    plt.subplot(2, 3, 1)
    plt.plot(history['x'], history['y'], 'b', label='Actual')
    if USE_STRAIGHT_TEST:
        plt.plot([0, 30], [0, 0], 'r--', label='Desired')
    else:
        th = np.linspace(0, 2 * np.pi, 200)
        plt.plot(center[0] + radius * np.cos(th),
                 center[1] + radius * np.sin(th), 'r--', label='Desired')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Tracking (World-Frame PID)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # 航向角跟踪
    plt.subplot(2, 3, 2)
    plt.plot(t_arr, history['phi'], 'b', label='Actual yaw')
    plt.plot(t_arr, history['phid'], 'r--', label='phi_d = atan2(u_G)')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.title('Heading Tracking')
    plt.legend()
    plt.grid(True)

    # 世界系误差
    plt.subplot(2, 3, 3)
    plt.plot(t_arr, history['err_x'], 'r', label='err X (world)')
    plt.plot(t_arr, history['err_y'], 'b', label='err Y (world)')
    plt.plot(t_arr, history['err_norm'], 'k', label='norm')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('World-Frame Tracking Error')
    plt.legend()
    plt.grid(True)

    # 航向偏移
    plt.subplot(2, 3, 4)
    plt.plot(t_arr, np.degrees(ephi_arr), 'c')
    plt.xlabel('Time (s)')
    plt.ylabel('ephi (deg)')
    plt.title('Heading Offset ephi')
    plt.grid(True)

    # 角速度: 指令 vs 实际
    plt.subplot(2, 3, 5)
    plt.plot(t_arr, history['w_cmd'], 'm', label='w_cmd')
    plt.plot(t_arr, history['wz'], 'k--', label='wz (meas)')
    plt.xlabel('Time (s)')
    plt.ylabel('rad/s')
    plt.title('Yaw Rate: cmd vs meas')
    plt.legend()
    plt.grid(True)

    # 力矩饱和检查
    plt.subplot(2, 3, 6)
    plt.plot(t_arr, tau_arr, 'r')
    plt.axhline(tau_max, color='k', linestyle='--', label='limit')
    plt.xlabel('Time (s)')
    plt.ylabel('max |tau| (Nm)')
    plt.title('Wheel Torque Saturation')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
