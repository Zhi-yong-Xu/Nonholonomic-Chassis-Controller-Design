import numpy as np
import mujoco


class MuJoCoObstacleSensor:


    def __init__(self, model, data, robot_body: str = "base_link",
                 distmax: float = 3.0, robot_radius: float = 0.6):
        self.model = model
        self.data = data
        self.distmax = distmax
        self.robot_body = robot_body
        self.robot_radius = float(robot_radius)   # 车体等效圆盘半径 r_c [m]

        # ---- ① 收集机器人本体子树内的所有 geom id（用于排除）----
        robot_body_id = model.body(robot_body).id
        self.self_geoms = set()
        for b in range(model.nbody):
            if self._is_descendant(model, b, robot_body_id):
                for g in range(model.body_geomadr[b],
                               model.body_geomadr[b] + model.body_geomnum[b]):
                    self.self_geoms.add(g)

        # ---- 扫描其余 geom，提取障碍的解析几何参数 ----
        # 说明：本工程中障碍 geom 均直接挂在世界体 worldbody 下，
        #       geom_pos 即世界坐标，geom_quat 为单位四元数（静置、不旋转）。
        self.obstacles = []   # 每项: dict(type, cx, cy, cz, R, half_len)
        for g in range(model.ngeom):
            if g in self.self_geoms:
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if model.geom_contype[g] == 0:
                continue

            gtype = model.geom_type[g]
            pos = model.geom_pos[g].copy()      # 世界坐标（世界体直挂）
            size = model.geom_size[g].copy()

            if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                # size = [半径, 半长]，圆柱轴沿局部 z（世界 z，未旋转）
                R, half_len = float(size[0]), float(size[1])
                self.obstacles.append(dict(
                    type="cylinder", cx=pos[0], cy=pos[1], cz=pos[2],
                    R=R, half_len=half_len))
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                # size = [半径]
                R = float(size[0])
                self.obstacles.append(dict(
                    type="sphere", cx=pos[0], cy=pos[1], cz=pos[2],
                    R=R, half_len=0.0))
            else:
                # 其他类型（box/mesh 等）不支持解析，跳过并提示
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                print(f"[障碍识别] 警告: geom {name} 类型不支持解析判距，已跳过")

        # ---- ③ 启动自检输出 ----
        print(f"[障碍识别] 机器人本体 geom 数 = {len(self.self_geoms)}")
        print(f"[障碍识别] 车体等效半径 r_c = {self.robot_radius:.3f} m")
        print(f"[障碍识别] 解析识别障碍 {len(self.obstacles)} 个:")
        for ob in self.obstacles:
            print(f"  {ob['type']:<9s} center=({ob['cx']:>6.2f},"
                  f"{ob['cy']:>6.2f},{ob['cz']:>6.2f}) R={ob['R']:.3f}")

    @staticmethod
    def _is_descendant(model, bid: int, ancestor_id: int) -> bool:
        while bid != 0:
            if bid == ancestor_id:
                return True
            bid = model.body_parentid[bid]
        return False

    # ---------- 运行时查询 ----------
    def nearest_point(self):

        px, py = self.data.qpos[0], self.data.qpos[1]
        pz = self._robot_surface_z()

        dmin = np.inf
        pmin = None
        cmin = None

        for ob in self.obstacles:
            dx, dy = px - ob["cx"], py - ob["cy"]
            dist_c = np.hypot(dx, dy)               # 到轴/球心的水平距离

            if ob["type"] == "cylinder":
                # 水平截圆半径 = R（含端面圆盘投影的简化处理）
                r2d = ob["R"]
            else:  # sphere
                dz = abs(pz - ob["cz"])
                if dz < ob["R"]:
                    # 球在机器人高度的水平截圆半径
                    r2d = np.sqrt(ob["R"] ** 2 - dz ** 2)
                else:
                    r2d = 1e-3   # 高度错开，几乎不相交

            # ★ 间隙距离：扣除车体等效圆盘半径 r_c
            d = dist_c - r2d - self.robot_radius

            if d < dmin:
                dmin = d
                # 表面最近点 = 圆心沿 (机器人-圆心) 方向移动 r2d
                if dist_c > 1e-9:
                    ux, uy = dx / dist_c, dy / dist_c
                else:
                    ux, uy = 1.0, 0.0   # 重合时取任意方向
                pmin = np.array([[ob["cx"] + r2d * ux],
                                 [ob["cy"] + r2d * uy]])
                cmin = np.array([[ob["cx"]], [ob["cy"]]])

        if pmin is None or dmin > self.distmax:
            return None, np.inf, None
        return pmin, float(dmin), cmin

    def _robot_surface_z(self) -> float:
        """机器人底盘高度（base_link 原点 z）。"""
        return float(self.data.qpos[2])

    def reset(self, data):
        self.data = data
