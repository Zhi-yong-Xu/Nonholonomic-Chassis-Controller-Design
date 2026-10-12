# -*- coding: utf-8 -*-
"""
检测 Husky 最小包络圆柱
  1) 高度 H : 从车顶/车底远处打竖直射线扫描 (起点在车外, 可靠)
  2) 半径 R : 从车外探测球面沿径向向轴线打水平射线,
"""
import mujoco
from mujoco import mjtObj
import numpy as np

# ---------------- 0. 加载模型 ----------------
model = mujoco.MjModel.from_xml_path(".\model\husky.xml")
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

base_id  = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "base_link")
base_pos = data.xpos[base_id].copy()          # base_link 世界坐标

_ray_geomid = np.zeros(1, dtype=np.int32)

def ray_cast(pnt, vec):
    pnt = np.asarray(pnt, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    return mujoco.mj_ray(model, data, pnt, vec, None, 0, -1, _ray_geomid)


# ---------------- 1. 检测高度范围 H ----------------
def detect_z_extent(xy_half=1.0, n_grid=60, z_far=3.0):
    zs_low, zs_high = [], []
    for x in np.linspace(-xy_half, xy_half, n_grid):
        for y in np.linspace(-xy_half, xy_half, n_grid):
            px, py = base_pos[0] + x, base_pos[1] + y
            d = ray_cast([px, py, base_pos[2] - z_far], [0.0, 0.0, 1.0])
            if d >= 0:
                zs_low.append(base_pos[2] - z_far + d)
            d = ray_cast([px, py, base_pos[2] + z_far], [0.0, 0.0, -1.0])
            if d >= 0:
                zs_high.append(base_pos[2] + z_far - d)
    if not zs_low or not zs_high:
        raise RuntimeError("射线未命中车体, 请增大 xy_half")
    return min(zs_low), max(zs_high)


zmin_w, zmax_w = detect_z_extent()
zmin = zmin_w - base_pos[2]                   # 转 base_link 系
zmax = zmax_w - base_pos[2]
print(f"高度包络: zmin = {zmin:+.4f} m, zmax = {zmax:+.4f} m, "
      f"H = {zmax - zmin:.4f} m")

# ---------------- 2. 检测半径 R (从外向轴心打射线) ----------------
R_PROBE = 1.5                        # 探测半径, 必须大于车体最大径向范围
N_THETA, N_Z = 720, 200              # 圆周 / 高度采样密度
zs     = np.linspace(zmin, zmax, N_Z)
thetas = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)

def max_radial_extent():
    """所有方向上车身径向范围的最大值 = 最小包络圆柱半径"""
    rho_max, best = 0.0, None
    for z in zs:
        pz = base_pos[2] + z
        for th in thetas:
            ct, st = np.cos(th), np.sin(th)
            # 探针在半径 R_PROBE 处, 射线指向轴心
            d = ray_cast([base_pos[0] + R_PROBE * ct,
                          base_pos[1] + R_PROBE * st,
                          pz], [-ct, -st, 0.0])
            if d >= 0:
                rho = R_PROBE - d    # 该 (theta, z) 方向车身最远点的半径
                if rho > rho_max:
                    rho_max, best = rho, (rho, np.degrees(th), z)
    return rho_max, best

R, best = max_radial_extent()
print(f"粗采样完成: R ≈ {R:.4f} m (命中角度 {best[1]:.2f}°, z = {best[2]:+.4f})")

# ---------------- 3. 角度局部细化 (精确定位最远点) ----------------
rho_best, th_best, z_best = best
th_lo, th_hi = np.radians(th_best - 360.0 / N_THETA), np.radians(th_best + 360.0 / N_THETA)
ct, st = np.cos(th_best), np.sin(th_best)
z_candidates = [z_best + dz for dz in (-0.002, 0.0, 0.002) if zmin <= z_best + dz <= zmax]

for _ in range(40):                  # 在最佳角度附近黄金分割细化
    t1 = th_lo + 0.382 * (th_hi - th_lo)
    t2 = th_lo + 0.618 * (th_hi - th_lo)
    vals = []
    for t in (t1, t2):
        c, s = np.cos(t), np.sin(t)
        dmax = -1
        for z in z_candidates:
            d = ray_cast([base_pos[0] + R_PROBE * c, base_pos[1] + R_PROBE * s,
                          base_pos[2] + z], [-c, -s, 0.0])
            if d >= 0:
                dmax = max(dmax, R_PROBE - d)
        vals.append(dmax)
    if vals[0] >= vals[1]:
        th_hi, rho_best = t2, max(rho_best, vals[0])
    else:
        th_lo, rho_best = t1, max(rho_best, vals[1])
R = rho_best

# ---------------- 4. 输出结果 ----------------
zc = 0.5 * (zmin + zmax)
H  = zmax - zmin
print("\n========== 最小包络圆柱(轴 = base_link z 轴) ==========")
print(f"半径  R      = {R:.4f} m")
print(f"高度  H      = {H:.4f} m")
print(f"z 范围       = [{zmin:+.4f}, {zmax:+.4f}] m (相对 base_link)")
print(f"轴向中心 zc  = {zc:+.4f} m")
