"""grasp_policy 单元测试（M1-1）：z_min、差分 IK 收敛、apply 硬约束与单步精度。
不依赖 ROS，直接通过相对路径定位场景，用 .venv-fr3 运行：
  source .venv-fr3/bin/activate && pytest franka_ros2_ws/src/fr3_vision_grasp/test/test_grasp_policy.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'franka_ros2_ws/src/fr3_vision_grasp'))

from fr3_vision_grasp.core import HOME
from fr3_vision_grasp.grasp_policy import (CartesianController, GraspConfig,
    finger_clearance, orientation_error, target_rotation)
from fr3_vision_grasp.simulation import Simulation

SCENE = ROOT / 'mujoco_simulations_fr3' / 'scene_grasp.xml'


@pytest.fixture(scope='module')
def sim():
    s = Simulation(SCENE)
    mujoco.mj_forward(s.model, s.data)
    yield s
    s.close()


@pytest.fixture(scope='module')
def controller(sim):
    return CartesianController(sim.model, sim.data, GraspConfig(), HOME,
                               sim.qadr, sim.dadr, sim.actuators, sim.gripper, sim.limits)


def test_finger_clearance_and_z_min(sim, controller):
    # 实测（方案 4.2）：手指最低点在 TCP 下方 9.0 mm → z_min = 0.05+0.009+0.004
    mujoco.mj_forward(sim.model, sim.data)
    assert finger_clearance(sim.model, sim.data, GraspConfig()) == pytest.approx(-0.009, abs=2e-3)
    assert controller.z_min == pytest.approx(0.063, abs=2e-3)


def test_solve_ik_random_targets(sim, controller):
    """方案 M1 通过标准①：随机目标都能收敛，位置误差 < 1 mm、姿态误差 < 0.005 rad。
    目标按环境实际采样分布生成（z ≈ 方块中心 0.065 + 0.12，yaw 至多 ±45°，方案 4.4①⑥）。
    近基座 + 大 yaw 的目标可能所有解都贴近限位（余量 < 0.1 rad），按 4.4⑥ 属于
    "换种子→重采方块位置"的预期拒绝，不算失败，但拒绝率应 ≤ 10%。"""
    rng = np.random.default_rng(7)
    converged, rejected = 0, 0
    for _ in range(40):
        xy = np.array([rng.uniform(0.30, 0.67), rng.uniform(-0.23, 0.23)])
        z = rng.uniform(0.175, 0.195)  # 感知 z(≈0.065) + 0.12 交接高度 + 噪声
        yaw = rng.uniform(-np.pi / 4, np.pi / 4)
        pos = np.append(xy, z)
        q = None
        for _ in range(5):  # 连续 5 个种子失败才判不可解（方案 4.4⑥）
            seed = HOME + rng.normal(0.0, 0.1, 7)
            q = controller.solve_ik(sim.model, sim.data, pos, yaw, seed)
            if q is not None:
                break
        if q is None:
            rejected += 1
            continue
        converged += 1
        sim.data.qpos[sim.qadr] = q
        mujoco.mj_forward(sim.model, sim.data)
        err_pos = np.linalg.norm(sim.data.site_xpos[controller.tcp] - pos)
        rot = sim.data.site_xmat[controller.tcp].reshape(3, 3)
        err_rot = np.linalg.norm(orientation_error(rot, target_rotation(yaw)))
        assert err_pos < 1e-3
        assert err_rot < 5e-3
    assert converged >= 36, f'IK acceptance too low: {converged}/40 converged, {rejected} rejected'
    sim.reset()


def test_apply_hard_constraints(sim, controller):
    """命令位姿的硬约束：z 不低于 z_min，xy 裁剪到工作区内；闭合动作写对 ctrl。"""
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    controller.reset(sim.data)
    cfg = GraspConfig()
    down = np.array([0.0, 0.0, -1.0, 0.0, -1.0])
    for _ in range(80):  # 持续向下 80 步，命令 z 早已越界
        controller.apply(sim.model, sim.data, down)
    assert controller.cmd_pos[2] >= controller.z_min - 1e-12
    (x0, x1), (y0, y1) = cfg.workspace
    assert x0 <= controller.cmd_pos[0] <= x1
    assert y0 <= controller.cmd_pos[1] <= y1
    assert sim.data.ctrl[sim.gripper] == pytest.approx(0.0)  # action[4]=-1 → 全闭
    out = np.array([1.0, -1.0, 0.0, 0.0, 1.0])
    for _ in range(400):  # x 正向越界、y 负向越界
        controller.apply(sim.model, sim.data, out)
    assert controller.cmd_pos[0] <= x1 + 1e-12
    assert controller.cmd_pos[1] >= y0 - 1e-12
    assert sim.data.ctrl[sim.gripper] == pytest.approx(0.04)  # action[4]=+1 → 全开
    sim.reset()


def test_apply_ik_single_step(sim, controller):
    """方案 M1 通过标准①的"单步笛卡尔误差"：差分 IK 一步解的线性化残差 < 1 mm。
    把 ctrl 目标直接设为 qpos（跳过动力学），检验运动学层面的单步精度。"""
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    controller.reset(sim.data)
    rng = np.random.default_rng(3)
    for _ in range(10):
        action = rng.uniform(-1.0, 1.0, 5)
        controller.apply(sim.model, sim.data, action)
        sim.data.qpos[sim.qadr] = sim.data.ctrl[sim.actuators]
        mujoco.mj_forward(sim.model, sim.data)
        err = np.linalg.norm(sim.data.site_xpos[controller.tcp] - controller.cmd_pos)
        assert err < 1e-3, f'single-step IK error {err * 1000:.2f} mm'
    sim.reset()


def test_apply_servo_tracking(sim, controller):
    """物理层跟踪：位置伺服 kp=4500/kv=450 过阻尼，时间常数 ≈ kv/kp = 0.1 s，
    一个 5 mm 策略步（50 ms）只能跟踪一部分（实测 ≈ 3.8 mm 残差）。
    该滞后是伺服固有特性，策略观测的是真实 TCP 状态，可自行补偿；
    此处只做量级回归（< 8 mm），防止控制器参数意外变化。"""
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    controller.reset(sim.data)
    rng = np.random.default_rng(5)
    for _ in range(5):
        action = rng.uniform(-1.0, 1.0, 5)
        controller.apply(sim.model, sim.data, action)
        for _ in range(25):
            sim.step()
        err = np.linalg.norm(sim.data.site_xpos[controller.tcp] - controller.cmd_pos)
        assert err < 8e-3, f'servo tracking error {err * 1000:.2f} mm'
    sim.reset()


def make_ctx(sim):
    from fr3_vision_grasp.grasp_policy import EpisodeCtx
    cfg = GraspConfig()
    cube = sim.model.body('red_object').id
    return EpisodeCtx(tcp=sim.tcp, fqadr=sim.fqadr, fdadr=sim.fdadr, gripper=sim.gripper,
                      cube_body=cube, cube_dadr=sim.model.jnt_dofadr[sim.model.body_jntadr[cube]],
                      finger_bodies=(sim.model.body('left_finger').id,
                                     sim.model.body('right_finger').id),
                      cfg=cfg)


def test_build_obs_dims_and_content(sim):
    """19 维观测；相对位置/步数比例等关键字段取值正确。"""
    from fr3_vision_grasp.grasp_policy import build_obs, privileged_obs
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    ctx = make_ctx(sim)
    ctx.begin_episode(perceived=np.array([0.45, 0.02, 0.065]), perceived_yaw=0.1, start_z=0.065)
    ctx.step, ctx.last_action = 12, np.full(5, 0.25)
    obs = build_obs(sim.model, sim.data, ctx)
    assert obs.shape == (19, )
    tcp = sim.data.site_xpos[sim.tcp]
    np.testing.assert_allclose(obs[:3], ctx.perceived - tcp, atol=1e-12)
    assert obs[18] == pytest.approx(12 / GraspConfig().max_steps)
    np.testing.assert_allclose(obs[13:18], ctx.last_action, atol=1e-12)
    assert obs[12] == pytest.approx(float(sim.data.actuator_force[sim.gripper]))
    priv = privileged_obs(sim.model, sim.data, ctx)
    assert priv.shape == (12, )
    np.testing.assert_allclose(priv[:3], sim.data.xpos[ctx.cube_body] - tcp, atol=1e-12)


def test_proprio_done_success_and_false_positive(sim, controller):
    """成功判据四条件：真夹持（指宽=方块尺寸、命令全闭）+ 曾下降到抓取高度
    （min_tcp_z ≤ start_z + grasp_z_tol）后抬升，连续 6 步判 success；
    空闭合（指宽精确跟踪命令值）即使抬得再高也绝不判 success（判据③防假成功）；
    没下降过就在空中闭合的真夹持同样不判 success（判据① A4 修复防假成功）；
    掉落（指宽塌陷）不判 success；碰桌/越界即时安全中止。"""
    from fr3_vision_grasp.grasp_policy import proprio_done
    cfg = GraspConfig()
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    # 把 TCP 放到贴近真实回合的高度：HOME 的 0.549 > z_max=0.50 会先触发越界中止，
    # 而真实回合起始 ≈ 0.185。0.20 是"没下去抓过就在空中闭合"的典型假成功场景。
    q = None
    rng = np.random.default_rng(4)
    for _ in range(10):
        seed = HOME + rng.normal(0.0, 0.1, 7)
        q = controller.solve_ik(sim.model, sim.data, np.array([0.45, 0.0, 0.20]), 0.0, seed)
        if q is not None:
            break
    assert q is not None
    sim.data.qpos[sim.qadr] = q
    sim.data.ctrl[sim.actuators] = q
    mujoco.mj_forward(sim.model, sim.data)
    ctx = make_ctx(sim)
    ctx.begin_episode(perceived=np.array([0.45, 0.0, 0.065]), perceived_yaw=0.0, start_z=0.065)
    # begin_episode 之后 min_tcp_z 重置为 inf
    assert ctx.min_tcp_z == np.inf
    tcp_z = float(sim.data.site_xpos[sim.tcp][2])
    assert tcp_z - 0.065 >= cfg.lift_height

    def set_gripper(qpos_per_finger, ctrl_per_finger):
        sim.data.qpos[sim.fqadr] = qpos_per_finger
        sim.data.ctrl[sim.gripper] = ctrl_per_finger
        mujoco.mj_forward(sim.model, sim.data)

    # 空闭合：指宽 == 命令宽（误差 0 < 3 mm）→ 永不 success
    set_gripper(0.010, 0.010)
    for _ in range(10):
        ctx.step += 1
        assert proprio_done(sim.model, sim.data, ctx) is None
    assert ctx.consecutive == 0

    # 真夹持但没下降过（min_tcp_z ≈ 0.20 > start_z + grasp_z_tol = 0.095）
    # → 不判 success（A4 修复：判据①要求"从本回合最低点抬升"）
    set_gripper(0.015, 0.0)
    for _ in range(10):
        ctx.step += 1
        assert proprio_done(sim.model, sim.data, ctx) is None
    assert ctx.consecutive == 0

    # 相当于已下降到 0.065 抓过：真夹持 + 从最低点抬升 0.135 ≥ 0.10，
    # 连续 6 步后 success
    ctx.min_tcp_z = 0.065
    result = None
    for _ in range(6):
        ctx.step += 1
        result = proprio_done(sim.model, sim.data, ctx)
    assert result == 'success'

    # 掉落：指宽塌陷到 0.005（< 0.5×0.03）→ 不再 success
    ctx.begin_episode(np.array([0.45, 0.0, 0.065]), 0.0, 0.065)
    ctx.min_tcp_z = 0.065
    set_gripper(0.0025, 0.0)
    for _ in range(10):
        ctx.step += 1
        assert proprio_done(sim.model, sim.data, ctx) is None

    # 超时：真夹持先积累 5 步，第 6 步恰好到达步数上限 → success 优先于 timeout
    ctx.begin_episode(np.array([0.45, 0.0, 0.065]), 0.0, 0.065)
    ctx.min_tcp_z = 0.065
    set_gripper(0.015, 0.0)
    for _ in range(cfg.success_hold - 1):
        ctx.step += 1
        assert proprio_done(sim.model, sim.data, ctx) is None
    ctx.step = cfg.max_steps
    assert proprio_done(sim.model, sim.data, ctx) == 'success'
    ctx.consecutive = 0
    set_gripper(0.010, 0.010)  # 空闭合 + 步数到顶 → timeout
    assert proprio_done(sim.model, sim.data, ctx) == 'timeout'
    sim.reset()


def test_proprio_done_safety_aborts(sim, controller):
    """越界与碰桌即时中止。apply 的 z_min 硬约束下手指与桌面永远保有
    z_margin=4 mm 间隙（M1-1 已验证防护有效），正常控制路径碰不到桌面；
    因此这里用 solve_ik（只有物理限位约束，无 z_min）直接解一个指尖压入
    桌面 2 mm 的位形，验证真实接触下检测即时触发、且持续下压时不消失。"""
    from fr3_vision_grasp.grasp_policy import proprio_done
    sim.reset()
    # 把方块移出工作区，避免干扰
    sim.set_free_body('red_object', np.array([0.90, 0.20, 0.065]))
    sim.set_free_body('blue_object', np.array([0.90, -0.20, 0.065]))
    sim.set_free_body('green_object', np.array([0.90, 0.0, 0.065]))
    # 指尖最低点在 TCP 下方 9 mm（finger_clearance 实测）；
    # TCP z=0.057 → 指尖 0.048，压入桌面(顶面 0.05) 2 mm
    q = None
    rng = np.random.default_rng(1)
    for _ in range(10):
        seed = HOME + rng.normal(0.0, 0.1, 7)
        q = controller.solve_ik(sim.model, sim.data, np.array([0.45, 0.0, 0.057]), 0.0, seed)
        if q is not None:
            break
    assert q is not None
    sim.data.qpos[sim.qadr] = q
    sim.data.ctrl[sim.actuators] = q
    sim.data.ctrl[sim.gripper] = 0.02
    mujoco.mj_forward(sim.model, sim.data)
    ctx = make_ctx(sim)
    ctx.begin_episode(np.array([0.45, 0.0, 0.065]), 0.0, 0.065)
    ctx.step += 1
    # mj_forward 已算出接触，检测应即时触发
    assert proprio_done(sim.model, sim.data, ctx) == 'table_contact'
    # 位置伺服维持下压：真实步进 0.5 s，接触持续、中止状态不消失
    for _ in range(10):
        for _ in range(25):
            sim.step()
        ctx.step += 1
        assert proprio_done(sim.model, sim.data, ctx) == 'table_contact'
    sim.reset()


def test_numpy_policy_forward(tmp_path):
    """NumpyPolicy 前向与手工计算一致；观测归一化、裁剪和动作裁剪都生效。"""
    from fr3_vision_grasp.grasp_policy import NumpyPolicy
    rng = np.random.default_rng(2)
    dims = [19, 32, 16]
    weights, biases = [], []
    for i in range(len(dims) - 1):
        weights.append(rng.normal(0, 0.5, (dims[i + 1], dims[i])))
        biases.append(rng.normal(0, 0.1, dims[i + 1]))
    action_w = rng.normal(0, 0.5, (5, dims[-1]))
    action_b = rng.normal(0, 0.1, 5)
    mean = rng.normal(0, 0.3, 19)
    var = rng.uniform(0.2, 1.5, 19)
    path = tmp_path / 'policy.npz'
    np.savez(path, pi_w0=weights[0], pi_b0=biases[0], pi_w1=weights[1], pi_b1=biases[1],
             action_w=action_w, action_b=action_b, obs_mean=mean, obs_var=var, clip_obs=3.0)
    policy = NumpyPolicy(path)
    obs = rng.normal(0, 1, 19)
    action = policy.act(obs)
    # 手工前向
    x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -3.0, 3.0)
    for w, b in zip(weights, biases):
        x = np.tanh(x @ w.T + b)
    expected = np.clip(x @ action_w.T + action_b, -1.0, 1.0)
    np.testing.assert_allclose(action, expected, atol=1e-10)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    # 极端观测被归一化裁剪截断，输出保持有限
    extreme = policy.act(np.full(19, 1e6))
    assert np.all(np.isfinite(extreme))
