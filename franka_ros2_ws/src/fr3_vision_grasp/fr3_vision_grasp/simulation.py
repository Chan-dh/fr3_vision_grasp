"""MuJoCo state owner. Rendering and stepping are called from one thread."""
import numpy as np
import mujoco
from contextlib import nullcontext
from .core import ARM, HOME, intrinsics


class Simulation:
    def __init__(self, scene, width=640, height=480):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(n).id for n in ARM]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.dadr = self.model.jnt_dofadr[self.joints]
        self.actuators = [self.model.actuator(n).id for n in ARM]
        self.finger_joints = [self.model.joint(f'finger_joint{i}').id for i in (1, 2)]
        self.fqadr = self.model.jnt_qposadr[self.finger_joints]
        self.fdadr = self.model.jnt_dofadr[self.finger_joints]
        self.gripper = self.model.actuator('gripper').id
        # RL 控制器使用的 TCP site（与 URDF 的 fr3_hand_tcp 一致，见 fr3.xml）
        self.tcp = self.model.site('gripper').id
        self.limits = self.model.jnt_range[self.joints].copy()
        self.camera = self.model.camera('top_camera').id
        self.width, self.height = width, height
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
        self.model.vis.quality.offsamples = 0
        self.model.vis.quality.shadowsize = 1024
        self.k = intrinsics(width, height, self.model.cam_fovy[self.camera])
        self.renderer = None
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = HOME
        self.data.qpos[self.fqadr] = 0.04
        self.data.ctrl[self.actuators] = HOME
        self.data.ctrl[self.gripper] = 0.04
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        # Compensate modeled bias, but preserve contact and gripper physics.
        self.data.qfrc_applied[self.dadr] = self.data.qfrc_bias[self.dadr]
        mujoco.mj_step(self.model, self.data)

    def render(self, lock=None):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.renderer.disable_depth_rendering()
        # Copy the scene under the physics lock, then render outside it. OpenGL
        # belongs to this caller thread; a slow GPU must not stall joint control.
        with lock if lock is not None else nullcontext():
            self.renderer.update_scene(self.data, camera=self.camera)
            self.frame_time = float(self.data.time)
        # Shadows/reflections are cosmetic and expensive on software OpenGL.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()  # Renderer returns linear, metric depth.
        self.renderer.disable_depth_rendering()
        far = self.model.vis.map.zfar * self.model.stat.extent
        depth[(depth >= far * 0.999) | ~np.isfinite(depth) | (depth <= 0)] = np.nan
        return rgb, depth.astype(np.float32)

    def camera_pose(self):
        rotation = self.data.cam_xmat[self.camera].reshape(3, 3) @ np.diag([1., -1., -1.])
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, rotation.ravel())
        return self.data.cam_xpos[self.camera].copy(), q

    def set_free_body(self, name, position, yaw=0.0):
        """设置带单个 freejoint 的 body 的位姿并清零速度，用于 RL reset。
        调用后需执行 mj_forward 才会更新派生量。"""
        bid = self.model.body(name).id
        if self.model.body_jntnum[bid] != 1 or self.model.jnt_type[self.model.body_jntadr[bid]] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f'body {name} does not carry a single freejoint')
        jid = self.model.body_jntadr[bid]
        qadr = self.model.jnt_qposadr[jid]
        self.data.qpos[qadr:qadr + 3] = position
        mujoco.mju_axisAngle2Quat(self.data.qpos[qadr + 3:qadr + 7],
                                  np.array([0.0, 0.0, 1.0]), yaw)
        self.data.qvel[self.model.jnt_dofadr[jid]:self.model.jnt_dofadr[jid] + 6] = 0.0

    def object_state(self, name):
        """带单个 freejoint 的方块真值（仅训练奖励/评估使用，部署端拿不到）：
        返回 (位置, yaw, 线速度, 角速度)，均为世界系。"""
        bid = self.model.body(name).id
        jid = self.model.body_jntadr[bid]
        qadr = self.model.jnt_qposadr[jid]
        dadr = self.model.jnt_dofadr[jid]
        pos = self.data.qpos[qadr:qadr + 3].copy()
        quat = self.data.qpos[qadr + 3:qadr + 7]
        yaw = float(np.arctan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                               1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)))
        return pos, yaw, self.data.qvel[dadr:dadr + 3].copy(), self.data.qvel[dadr + 3:dadr + 6].copy()

    def contacts(self, object_name):
        body = self.model.body(object_name).id
        fingers = {self.model.body(n).id: n for n in ('left_finger', 'right_finger')}
        sides = set()
        for contact in self.data.contact:
            a, b = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if a == body and b in fingers:
                sides.add(fingers[b])
            if b == body and a in fingers:
                sides.add(fingers[a])
        return sorted(sides)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
