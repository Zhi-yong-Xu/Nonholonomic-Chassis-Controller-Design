"""
Husky 转向死区测试 (MuJoCo)
原理: 差速力矩从0逐渐增大, 检测底盘开始产生偏航角速度的最小力矩
用法: python husky_deadzone_test.py
"""
import mujoco
import numpy as np

# ---------------- 参数 ----------------
XML_PATH        = r".\model\husky.xml"    # 你的MJCF文件路径
TORQUE_MAX      = 20.0           # 最大测试力矩 (N·m)
TORQUE_STEP     = 0.1            # 力矩递增步长 (N·m)
HOLD_STEPS      = 1000           # 每级力矩保持的仿真步数
YAW_RATE_THRESH = 0.02           # 判定"开始转动"的偏航角速度阈值 (rad/s)
SETTLE_STEPS    = 500            # 每级开始前的稳定步数(不计入统计)
SIM_DT          = None           # 从模型读取

# ---------------- 加载模型 ----------------
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
SIM_DT = model.opt.timestep

assert model.nu == 4, "期望4个轮电机执行器"

WHEEL_JOINTS = ["front_left_wheel", "front_right_wheel",
                "rear_left_wheel",  "rear_right_wheel"]
jnt_ids = [model.joint(name).id for name in WHEEL_JOINTS]

BASE_ID = model.body("base_link").id


def quat_to_yaw(q):
    """四元数 (w,x,y,z) -> 偏航角 yaw (rad), ZYX欧拉角约定"""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def set_diff_torque(tau):
    """差速力矩: 左轮 +tau, 右轮 -tau (原地自旋)"""
    data.ctrl[:] = [tau, -tau, tau, -tau]


def run_level(tau):
    """施加力矩tau, 返回稳态平均偏航角速度 (rad/s)"""
    mujoco.mj_resetData(model, data)          # 每级重置, 保证独立
    data.qpos[2] = 0.2                        # 初始高度(与xml一致)
    mujoco.mj_forward(model, data)

    # 让车先落稳
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    yaw0   = quat_to_yaw(data.xquat[BASE_ID])
    yaw_sum = 0.0
    prev_yaw = yaw0

    for _ in range(HOLD_STEPS):
        set_diff_torque(tau)
        mujoco.mj_step(model, data)
        yaw = quat_to_yaw(data.xquat[BASE_ID])
        dyaw = yaw - prev_yaw
        # 角度解卷绕
        if dyaw >  np.pi: dyaw -= 2 * np.pi
        if dyaw < -np.pi: dyaw += 2 * np.pi
        yaw_sum += dyaw
        prev_yaw = yaw

    yaw_rate = yaw_sum / (HOLD_STEPS * SIM_DT)
    # 左右轮平均转速(参考)
    wheel_vel = [data.qvel[model.jnt_dofadr[j]] for j in jnt_ids]
    return yaw_rate, wheel_vel


# ---------------- 死区扫描 ----------------
print(f"{'tau(Nm)':>10} | {'yaw_rate(rad/s)':>16} | {'FL':>8} {'FR':>8} {'RL':>8} {'RR':>8}")
print("-" * 70)

results = []
deadzone = None

for tau in np.arange(0.0, TORQUE_MAX + 1e-9, TORQUE_STEP):
    yaw_rate, wv = run_level(tau)
    results.append((tau, yaw_rate))
    print(f"{tau:10.2f} | {yaw_rate:16.5f} | "
          f"{wv[0]:8.3f} {wv[1]:8.3f} {wv[2]:8.3f} {wv[3]:8.3f}")

    if deadzone is None and abs(yaw_rate) > YAW_RATE_THRESH:
        deadzone = tau
        print(f"\n>>> 检测到转向死区阈值: tau ≈ {tau:.2f} N·m "
              f"(yaw_rate = {yaw_rate:.4f} rad/s 超过阈值 {YAW_RATE_THRESH})\n")

if deadzone is None:
    print(f"\n>>> 在 0 ~ {TORQUE_MAX} N·m 范围内未检测到死区 "
          f"(或阈值 {YAW_RATE_THRESH} rad/s 过高)")

# ---------------- 可视化 ----------------
try:
    import matplotlib.pyplot as plt
    taus, rates = zip(*results)
    plt.figure(figsize=(8, 5))
    plt.plot(taus, np.abs(rates), "b-o", markersize=3)
    plt.axvline(deadzone, color="r", linestyle="--",
                label=f"deadzone ≈ {deadzone:.2f} N·m" if deadzone else "no deadzone")
    plt.axhline(YAW_RATE_THRESH, color="g", linestyle=":",
                label=f"threshold {YAW_RATE_THRESH} rad/s")
    plt.xlabel("Differential torque (N·m)")
    plt.ylabel("|Yaw rate| (rad/s)")
    plt.title("Husky Steering Deadzone Test")
    plt.grid(True)
    plt.legend()
    plt.savefig("husky_deadzone.png", dpi=150)
    plt.show()
except ImportError:
    print("未安装matplotlib, 跳过绘图")
