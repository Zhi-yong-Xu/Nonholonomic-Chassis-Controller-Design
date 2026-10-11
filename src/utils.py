"""数学工具、MuJoCo 可视化辅助、MJCF 参数提取"""
import numpy as np
import mujoco


# ============================================================
# 数学工具
# ============================================================
def sig(x, p):
    """幂次符号函数 |x|^p * sign(x)"""
    return np.abs(x) ** p * np.sign(x)


def wrap_to_pi(x):
    """角度归一化到 (-pi, pi]"""
    return np.arctan2(np.sin(x), np.cos(x))


def quat_to_yaw(q):
    """四元数 -> 偏航角"""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ============================================================
# MJCF 参数提取
# ============================================================
def extract_husky_params(model) -> dict:
    """
    从 MJCF 模型中自动提取 Husky 运动学参数。
    MJCF 是唯一参数源（single source of truth）。

    对应关系：
      wheel_radius : 轮碰撞圆柱 geom 的 size[0]
      half_track   : 轮体相对 base_link 的 y 偏移（取左轮）
      wheel_base   : 前轮 x - 后轮 x
    """
    # ---- 轮半径：碰撞圆柱 geom 的 size[0] ----
    r = float(model.geom("front_left_wheel_col").size[0])

    # ---- 半轮距 / 轴距：轮体挂载位置 ----
    fl = model.body("front_left_wheel_link").pos    # [ 0.256,  0.2854, ...]
    rl = model.body("rear_left_wheel_link").pos     # [-0.256,  0.2854, ...]
    fr = model.body("front_right_wheel_link").pos   # [ 0.256, -0.2854, ...]

    half_track = float(fl[1])
    wheel_base = float(fl[0] - rl[0])

    # ---- 对称性校验 ----
    assert abs(fr[1] - (-half_track)) < 1e-9, "左右轮距不对称，检查 MJCF"
    assert abs(fr[0] - fl[0]) < 1e-9, "前后轮 x 不对齐，检查 MJCF"
    assert r > 0 and wheel_base > 0 and half_track > 0, "提取的运动学参数非法"

    return {
        "wheel_radius": r,
        "half_track":   half_track,
        "wheel_base":   wheel_base,
    }


def validate_params(mjcf_params: dict, cfg: dict, tol: float = 1e-6):
    """
    交叉校验：MJCF 提取的参数 vs YAML 中的参考值（husky_check 段）。
    YAML 中设为 null 或删除该段则跳过校验。
    """
    ref = cfg.get("husky_check", {}) or {}
    for key, val in ref.items():
        if val is None:
            continue
        if key not in mjcf_params:
            raise KeyError(f"未知校验参数 '{key}'，可用: {list(mjcf_params)}")
        if abs(mjcf_params[key] - val) > tol:
            raise ValueError(
                f"参数不一致: '{key}'  MJCF={mjcf_params[key]}, YAML={val}\n"
                f"请以 MJCF 为准更新 YAML，或检查模型文件。"
            )


# ============================================================
# MuJoCo 可视化辅助
# ============================================================
def add_line(scn, p1, p2, radius, rgba):
    """在 MuJoCo 场景中添加一条圆柱线段"""
    if scn.ngeom >= scn.maxgeom:
        return
    mid = (p1 + p2) / 2.0
    diff = p2 - p1
    length = np.linalg.norm(diff)
    if length < 1e-10:
        return
    direction = diff / length
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(direction, z_axis)) > 0.999:
        mat = np.eye(3) if np.dot(direction, z_axis) > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        v = np.cross(z_axis, direction)
        v = v / np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        s = np.sqrt(1.0 - c * c)
        mat = np.array([
            [c + v[0]*v[0]*(1-c),      v[0]*v[1]*(1-c) - v[2]*s, v[0]*v[2]*(1-c) + v[1]*s],
            [v[1]*v[0]*(1-c) + v[2]*s, c + v[1]*v[1]*(1-c),      v[1]*v[2]*(1-c) - v[0]*s],
            [v[2]*v[0]*(1-c) - v[1]*s, v[2]*v[1]*(1-c) + v[0]*s, c + v[2]*v[2]*(1-c)]
        ])
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, length / 2.0, 0.0]),
        mid, mat.flatten(), rgba
    )
    scn.ngeom += 1


def add_sphere(scn, pos, radius, rgba):
    """在 MuJoCo 场景中添加一个球体标记"""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        pos, np.eye(3).flatten(), rgba
    )
    scn.ngeom += 1
