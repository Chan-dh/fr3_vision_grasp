"""RL 抓取策略的共享模块：训练（rl/）与部署（bridge policy mode）都 import 这里。

只依赖 numpy 和 mujoco，不依赖 ROS、cv2、torch；
观测、动作、差分 IK、安全约束只在这一处实现，保证两边行为一致。
规格来源：docs/rl_grasp_plan.md 第 4 节。
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class GraspConfig:
    control_dt: float = 0.05            # 策略 20 Hz；物理步长 2 ms，每个策略步推进 25 个物理步
    max_dpos: float = 0.005             # 每步最多移动 5 mm
    max_dyaw: float = np.deg2rad(3.0)   # 每步最多转 3°
    table_z: float = 0.05
    z_margin: float = 0.004             # 3–5 mm；位置伺服跟踪会有少量超调，不能取 0
    workspace: tuple = ((0.25, 0.75), (-0.35, 0.35))
    z_max: float = 0.50                 # TCP 高度上界（越界判定用；正常操作 ≤ ~0.35）
    lift_height: float = 0.10
    grasp_z_tol: float = 0.03    # 判据①辅助：本回合 TCP 最低点须 ≤ start_z + 3 cm（证明"下去抓过"）
    max_steps: int = 120                # 6 s
    tcp_site: str = 'gripper'           # MJCF 中位于 hand 坐标系 0 0 0.1034，与 URDF 的 fr3_hand_tcp 一致
    grip_force: float = 20.0
    cube_size: float = 0.03
    grip_margin: float = 0.003          # 成功判据③：实际指宽 − 命令总宽度 ≥ 3 mm
    success_hold: int = 6               # 成功判据需连续保持的步数（6 步 ≈ 0.3 s）
    obs_dim: int = 19                   # build_obs 输出维度（文档见函数注释）
    action_dim: int = 5


def wrap_angle(a):
    """把角度卷绕到 [-pi, pi)。"""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def target_rotation(yaw):
    """TCP 目标姿态：工具 z 轴朝下（ROS 中 Quaternion(x=1)），再绕世界 z 轴转 yaw。"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0],
                     [s, -c, 0.0],
                     [0.0, 0.0, -1.0]])


def tcp_yaw(rot):
    """从 TCP 旋转矩阵提取绕世界 z 的 yaw（假设工具接近朝下）。"""
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def orientation_error(rot_cur, rot_des):
    """世界系下的姿态误差（3 维），配合 mj_jacSite 的角速度雅可比使用。"""
    return 0.5 * (np.cross(rot_cur[:, 0], rot_des[:, 0])
                  + np.cross(rot_cur[:, 1], rot_des[:, 1])
                  + np.cross(rot_cur[:, 2], rot_des[:, 2]))


def finger_clearance(model, data, cfg, fingers=('left_finger', 'right_finger')):
    """手指碰撞几何最低点相对 TCP 的 z 偏移（负值 = 在 TCP 下方）。
    需先 mj_forward。box 用精确半边长，mesh 用顶点，其余用 rbound 保守估计。
    实测（2026-10-05）：手指网格与 fingertip_pad 均为 TCP 下方 9.0 mm，开闭指不影响。"""
    sid = model.site(cfg.tcp_site).id
    tcp_z = data.site_xpos[sid][2]
    lowest = np.inf
    for name in fingers:
        bid = model.body(name).id
        start = model.body_geomadr[bid]
        for g in range(start, start + model.body_geomnum[bid]):
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue  # 纯视觉 geom 不参与碰撞
            p, r = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                z_low = p[2] - (np.abs(r) @ model.geom_size[g])[2]
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                if vn:
                    verts = model.mesh_vert[va:va + vn].reshape(-1, 3) @ r.T + p
                    z_low = verts[:, 2].min()
                else:
                    z_low = p[2] - model.geom_rbound[g]
            else:
                z_low = p[2] - model.geom_rbound[g]
            lowest = min(lowest, z_low)
    return lowest - tcp_z


class CartesianController:
    """把 5 维动作（Δx, Δy, Δz, Δyaw, 夹爪）换成关节位置伺服的 ctrl。
    "朝下"由控制器保证，策略只需输出增量；差分 IK 保证训练与部署共用同一套运动学。"""

    def __init__(self, model, data, cfg, q_home, arm_qadr, arm_dofs, arm_act,
                 gripper_act, joint_limits, lam=0.05, k_null=0.05, max_dq=0.2):
        self.cfg = cfg
        self.q_home = np.asarray(q_home, dtype=float)
        self.qadr = np.asarray(arm_qadr)
        self.dofs = np.asarray(arm_dofs)
        self.act = np.asarray(arm_act)
        self.gripper = int(gripper_act)
        self.limits = np.asarray(joint_limits, dtype=float)
        self.lam, self.k_null, self.max_dq = lam, k_null, max_dq
        self.tcp = model.site(cfg.tcp_site).id
        # z_min = 桌面 + 手指最低点在 TCP 下方的距离 + 余量（方案 4.2）
        self.finger_offset = finger_clearance(model, data, cfg)
        self.z_min = cfg.table_z - self.finger_offset + cfg.z_margin
        self.cmd_pos = None
        self.cmd_yaw = None

    # —— 差分 IK 核心：阻尼最小二乘 + 零空间拉回 home ——
    def _ik_step(self, model, data, err, q):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, self.tcp)
        jac = np.vstack([jacp, jacr])[:, self.dofs]                       # 6×7
        dq = jac.T @ np.linalg.solve(jac @ jac.T + self.lam ** 2 * np.eye(6), err)
        if self.k_null > 0:
            nullspace = np.eye(7) - np.linalg.pinv(jac) @ jac
            dq = dq + nullspace @ (self.k_null * (self.q_home - q))
        return dq

    def reset(self, data):
        """接管时调用：命令位姿 = 当前 TCP 位姿，从零误差开始。"""
        self.cmd_pos = data.site_xpos[self.tcp].copy()
        self.cmd_yaw = tcp_yaw(data.site_xmat[self.tcp].reshape(3, 3))

    def solve_ik(self, model, data, pos, yaw, q_seed, max_iter=200,
                 tol_pos=5e-4, tol_rot=5e-3, joint_margin=0.1, rot_des=None):
        """reset 前用：从 q_seed 迭代差分 IK 到目标位姿。成功返回关节位形（7,）。
        迭代中只受物理限位约束（搜索阶段裁到 margin 会挡住解路径，实测近基座
        低高度目标会因此全部失败）；收敛后按方案 4.4⑥ 验收：任一关节离限位
        < joint_margin 时返回 None，由调用方换种子重试。data.qpos 会被修改。
        rot_des 可传入扰动后的目标旋转（训练 reset 模拟 MoveIt 的 roll/pitch
        到达误差用）；缺省为精确的"朝下 + yaw"。"""
        q = np.clip(np.asarray(q_seed, dtype=float), self.limits[:, 0], self.limits[:, 1])
        rot_des = target_rotation(yaw) if rot_des is None else np.asarray(rot_des, dtype=float)
        for _ in range(max_iter):
            data.qpos[self.qadr] = q
            mujoco.mj_forward(model, data)
            e_pos = np.asarray(pos) - data.site_xpos[self.tcp]
            e_rot = orientation_error(data.site_xmat[self.tcp].reshape(3, 3), rot_des)
            if np.linalg.norm(e_pos) < tol_pos and np.linalg.norm(e_rot) < tol_rot:
                if (np.any(q < self.limits[:, 0] + joint_margin)
                        or np.any(q > self.limits[:, 1] - joint_margin)):
                    return None
                return q.copy()
            err = np.concatenate([e_pos, e_rot])
            dq = self._ik_step(model, data, err, q)
            step = np.linalg.norm(dq)
            if step > self.max_dq:
                dq *= self.max_dq / step
            q = np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])
        return None

    def apply(self, model, data, action):
        """每个策略步调用一次，写 ctrl；随后由调用方推进 25 个物理步。"""
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        cfg = self.cfg
        # 1) 命令位姿 += 增量，xy 裁剪到工作区，z 不低于 z_min（硬约束，方案 4.2 第 1 层）
        self.cmd_pos += action[:3] * cfg.max_dpos
        (x0, x1), (y0, y1) = cfg.workspace
        self.cmd_pos[0] = np.clip(self.cmd_pos[0], x0, x1)
        self.cmd_pos[1] = np.clip(self.cmd_pos[1], y0, y1)
        self.cmd_pos[2] = max(self.cmd_pos[2], self.z_min)
        self.cmd_yaw = wrap_angle(self.cmd_yaw + action[3] * cfg.max_dyaw)
        # 2) 误差 = 命令位姿 − 当前 TCP 位姿（位置 3 + 姿态 3，姿态目标为"朝下 + 命令 yaw"）
        e_pos = self.cmd_pos - data.site_xpos[self.tcp]
        e_rot = orientation_error(data.site_xmat[self.tcp].reshape(3, 3),
                                  target_rotation(self.cmd_yaw))
        # 3) 差分 IK：dq 写入 ctrl[arm] = q + dq；零空间已拉向 home
        q = data.qpos[self.qadr]
        dq = self._ik_step(model, data, np.concatenate([e_pos, e_rot]), q)
        data.ctrl[self.act] = np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])
        # 4) 夹爪：ctrl 0–0.04 是单指行程，action=−1 全闭、+1 全开
        data.ctrl[self.gripper] = (action[4] + 1.0) / 2.0 * 0.04


@dataclass
class EpisodeCtx:
    """回合上下文：env/bridge 在 reset 时创建。感知目标在整个回合中固定不变
    （俯拍相机在抓取时会被夹爪挡住，部署时拿不到实时方块位姿，方案第 2 节）。
    感知相关字段由调用方填入（训练 = 真值 + 噪声，部署 = perception 输出）。"""
    tcp: int                              # TCP site id
    fqadr: np.ndarray                     # 两指 qpos 地址
    fdadr: np.ndarray                     # 两指 qvel 地址
    gripper: int                          # 夹爪执行器 id
    cube_body: int = -1                   # 方块 body id（privileged 观测用）
    cube_dadr: int = -1                   # 方块 freejoint 的 qvel 起始地址
    finger_bodies: tuple = ()             # (left_id, right_id)，接触判定用
    table_body: str = 'table'
    cfg: GraspConfig = field(default_factory=GraspConfig)
    perceived: np.ndarray = None          # 感知目标位置（世界系）
    perceived_yaw: float = 0.0
    start_z: float = 0.065                # 起始方块中心高度，成功判据①的基准
    min_tcp_z: float = np.inf             # 本回合 TCP 最低点（判据①：从最低点抬升，防空中闭合假成功）
    step: int = 0
    consecutive: int = 0                  # 成功条件连续保持计数（消抖）
    last_action: np.ndarray = None

    def __post_init__(self):
        if self.last_action is None:
            self.last_action = np.zeros(self.cfg.action_dim)

    def begin_episode(self, perceived, perceived_yaw, start_z):
        """每个回合开始时重置回合内状态（id 等静态字段保持不变）。"""
        self.perceived = np.asarray(perceived, dtype=float)
        self.perceived_yaw = float(perceived_yaw)
        self.start_z = float(start_z)
        self.min_tcp_z = np.inf
        self.step = 0
        self.consecutive = 0
        self.last_action = np.zeros(self.cfg.action_dim)


def build_obs(model, data, ctx):
    """actor 的观测（19 维，只用部署时能拿到的量）：
    感知目标相对 TCP 的位置(3) + (感知yaw − TCP yaw) 的 sin/cos(2) + TCP 速度(3)
    + 两指位置和速度(4) + 夹爪执行器力(1) + 上一步动作(5) + 已执行步数比例(1)。"""
    cfg = ctx.cfg
    tcp_pos = data.site_xpos[ctx.tcp]
    yaw = tcp_yaw(data.site_xmat[ctx.tcp].reshape(3, 3))
    dyaw = wrap_angle(ctx.perceived_yaw - yaw)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, ctx.tcp)
    return np.concatenate([
        ctx.perceived - tcp_pos,                          # 3
        [np.sin(dyaw), np.cos(dyaw)],                     # 2
        jacp @ data.qvel,                                 # 3（世界系线速度）
        data.qpos[ctx.fqadr], data.qvel[ctx.fdadr],       # 4
        [float(data.actuator_force[ctx.gripper])],        # 1
        ctx.last_action,                                  # 5
        [ctx.step / cfg.max_steps],                       # 1
    ])


def cube_contacts(model, data, ctx):
    """方块与两指的接触侧集合（privileged：用了仿真接触真值）。"""
    sides = set()
    fingers = set(ctx.finger_bodies)
    for contact in data.contact[:data.ncon]:
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]
        if b1 == ctx.cube_body and b2 in fingers:
            sides.add(b2)
        if b2 == ctx.cube_body and b1 in fingers:
            sides.add(b1)
    return sides


def privileged_obs(model, data, ctx):
    """给 critic 的可选观测（12 维，含仿真真值；actor 永远不用，方案 4.3）：
    方块真实位置相对 TCP(3) + (方块yaw − TCP yaw) sin/cos(2) + 方块线速度(3)
    + 方块角速度 z(1) + 左右指接触(2) + 方块离桌面高度(1)。"""
    cfg = ctx.cfg
    tcp_pos = data.site_xpos[ctx.tcp]
    yaw = tcp_yaw(data.site_xmat[ctx.tcp].reshape(3, 3))
    pos = data.xpos[ctx.cube_body]
    quat = data.xquat[ctx.cube_body]
    cube_yaw = float(np.arctan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                                1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)))
    dyaw = wrap_angle(cube_yaw - yaw)
    vel = data.qvel[ctx.cube_dadr:ctx.cube_dadr + 3]
    angvel = data.qvel[ctx.cube_dadr + 3:ctx.cube_dadr + 6]
    sides = cube_contacts(model, data, ctx)
    return np.concatenate([
        pos - tcp_pos,                                    # 3
        [np.sin(dyaw), np.cos(dyaw)],                     # 2
        vel, [angvel[2]],                                 # 4
        [1.0 if ctx.finger_bodies[0] in sides else 0.0,
         1.0 if ctx.finger_bodies[1] in sides else 0.0],  # 2
        [pos[2] - cfg.table_z],                           # 1
    ])


def fingers_touch_table(model, data, ctx):
    """任一手指与桌面接触（安全中止判据；方块压在桌上不算，只查手指）。"""
    fingers = set(ctx.finger_bodies)
    for contact in data.contact[:data.ncon]:
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]
        n1, n2 = model.body(b1).name, model.body(b2).name
        if (b1 in fingers and n2 == ctx.table_body) or (b2 in fingers and n1 == ctx.table_body):
            return True
    return False


def proprio_done(model, data, ctx):
    """只用本体量判断的回合终止（部署时没有仿真真值）。返回
    'success' | 'out_of_bounds' | 'table_contact' | 'timeout' | None。
    success（方案 4.1，四条件同时满足且连续 6 步 / 0.3 s）：
      ① 从本回合 TCP 最低点抬升 ≥ lift_height，且最低点曾下降到
         start_z + grasp_z_tol 以内（开局 TCP 已在方块上方 12 cm，旧判据
         "tcp_z − start_z ≥ lift_height" 开局即满足，空中闭合会被判假成功）
      ② 实际总指宽 ∈ [0.5, 1.2] × 方块尺寸
      ③ 实际总指宽 − 命令总宽度 ≥ grip_margin（3 mm，防"空闭合假成功"：
         空闭合时指宽精确跟踪命令值，夹住方块时命令 0、实际 ≈ 方块尺寸）
      ④ 未触发碰桌、越界等安全中止（安全中止在本函数前面即时返回）"""
    cfg = ctx.cfg
    tcp = data.site_xpos[ctx.tcp]
    (x0, x1), (y0, y1) = cfg.workspace
    # 安全中止：越界（xy 出工作区 +0.01 容差、飞得太高）
    if not (x0 - 0.01 <= tcp[0] <= x1 + 0.01 and y0 - 0.01 <= tcp[1] <= y1 + 0.01
            and tcp[2] <= cfg.z_max):
        return 'out_of_bounds'
    # 安全中止：手指碰桌
    if fingers_touch_table(model, data, ctx):
        return 'table_contact'
    # 成功四条件
    width = float(data.qpos[ctx.fqadr].sum())          # 实际总指宽
    cmd_width = 2.0 * float(data.ctrl[ctx.gripper])    # 命令总宽度（ctrl 是单指行程）
    ctx.min_tcp_z = min(ctx.min_tcp_z, tcp[2])         # 安全中止之后才更新
    lifted = (tcp[2] - ctx.min_tcp_z >= cfg.lift_height
              and ctx.min_tcp_z <= ctx.start_z + cfg.grasp_z_tol)
    width_ok = 0.5 * cfg.cube_size <= width <= 1.2 * cfg.cube_size
    holding = width - cmd_width >= cfg.grip_margin
    if lifted and width_ok and holding:
        ctx.consecutive += 1
        if ctx.consecutive >= cfg.success_hold:
            return 'success'
    else:
        ctx.consecutive = 0
    if ctx.step >= cfg.max_steps:
        return 'timeout'
    return None


class NumpyPolicy:
    """加载 export_policy.py 导出的 npz（SB3 policy_net + action_net 权重，
    外加 VecNormalize 的 obs 统计），纯 numpy 前向；ROS 端不需要 torch。
    网络结构与 SB3 MlpExtractor 一致：tanh 激活的 MLP + 线性输出层，
    确定性动作并裁剪到 [-1, 1]。"""

    def __init__(self, path):
        d = np.load(path)
        self.obs_mean = d['obs_mean']
        self.obs_var = d['obs_var']
        self.clip_obs = float(d['clip_obs'])
        self.weights, self.biases = [], []
        i = 0
        while f'pi_w{i}' in d:
            self.weights.append(d[f'pi_w{i}'])
            self.biases.append(d[f'pi_b{i}'])
            i += 1
        if not self.weights:
            raise ValueError(f'{path}: no pi_w* arrays found')
        self.action_w = d['action_w']
        self.action_b = d['action_b']

    def act(self, obs):
        x = (np.asarray(obs, dtype=float) - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        x = np.clip(x, -self.clip_obs, self.clip_obs)
        for w, b in zip(self.weights, self.biases):
            x = np.tanh(x @ w.T + b)
        return np.clip(x @ self.action_w.T + self.action_b, -1.0, 1.0)
