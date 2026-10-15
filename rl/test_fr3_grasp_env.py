"""Fr3GraspEnv 冒烟测试（M1-3）：reset 按 4.4 规格复现、step 循环、课程与随机化。
方案 A（rl_reward_fix_plan.md A5）追加 PBRS 奖励性质测试：telescoping、
三类漏洞策略不划算、脚本基线显著占优、Φ 值域。
运行：source .venv-fr3/bin/activate && pytest rl/test_fr3_grasp_env.py -v -p no:cacheprovider
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'franka_ros2_ws/src/fr3_vision_grasp'))

from fr3_grasp_env import CUBES, Fr3GraspEnv
from scripted_baseline import scripted_actions


@pytest.fixture(scope='module')
def env():
    e = Fr3GraspEnv(stage='A')
    yield e
    e.sim.close()


def test_reset_initial_state(env):
    """4.4 规格复现：TCP 在感知目标上方 12 cm、夹爪全开、方块平放、远处干扰块。"""
    obs, info = env.reset(seed=42)
    assert obs.shape == (19,) and obs.dtype == np.float32
    tcp = env.sim.data.site_xpos[env.sim.tcp]
    perceived = env.ctx.perceived
    # 位置伺服 + 重力补偿下 0.2 s 稳定期后的残差应很小
    assert abs(tcp[2] - (perceived[2] + 0.12)) < 0.005
    assert np.linalg.norm(tcp[:2] - perceived[:2]) < 0.006
    # 夹爪全开（总开口 8 cm）
    assert env.sim.data.qpos[env.sim.fqadr].sum() == pytest.approx(0.08, abs=1e-3)
    # 目标方块：z=0.065、速度≈0（桌面软接触 settle 后有 <0.5 mm 下沉）、
    # 阶段 A yaw=0；感知噪声在截断界内
    pos, yaw, vel, _ = env.sim.object_state(env.target_name)
    assert pos[2] == pytest.approx(0.065, abs=1e-3)
    assert np.linalg.norm(vel) == pytest.approx(0.0, abs=1e-3)
    assert abs(yaw) < 1e-9
    assert np.linalg.norm(perceived[:2] - pos[:2]) <= 0.005 + 1e-9
    # 阶段 A：另两个方块移出工作区
    for name in CUBES:
        if name != env.target_name:
            assert env.sim.object_state(name)[0][0] == pytest.approx(0.90, abs=1e-6)
    assert info['stage'] == 'A'
    assert info['privileged'].shape == (12,)


def test_stage_a0_approach_height(env):
    """E1（rl_reward_fix_plan.md §5）：A0 档交接高度 6 cm，其余同 A。"""
    env.set_stage('A0')
    try:
        env.reset(seed=7)
        tcp = env.sim.data.site_xpos[env.sim.tcp]
        perceived = env.ctx.perceived
        assert abs(tcp[2] - (perceived[2] + 0.06)) < 0.005
        assert np.linalg.norm(tcp[:2] - perceived[:2]) < 0.006
    finally:
        env.set_stage('A')


def test_step_random_actions(env):
    """随机动作循环：观测/奖励有限，terminated/truncated 类型正确。"""
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 5).astype(np.float32))
        assert obs.shape == (19,) and np.all(np.isfinite(obs)) and np.isfinite(r)
        assert isinstance(term, (bool, np.bool_)) and isinstance(trunc, (bool, np.bool_))
        assert info['status'] is None or isinstance(info['status'], str)
        if term or trunc:
            env.reset(seed=2)


def test_timeout(env):
    """悬停不动直到步数上限：truncated=True、status='timeout'。"""
    env.reset(seed=3)
    hold = np.zeros(5, dtype=np.float32)
    status, ended = None, False
    for _ in range(env.cfg.max_steps):
        _, _, term, trunc, info = env.step(hold)
        if term or trunc:
            status, ended = info['status'], trunc
            break
    assert status == 'timeout' and ended


def test_stage_d_placement(env):
    """D 阶段：干扰方块与目标 ≥ min_dist，干扰方块之间不重叠。"""
    env.set_stage('D')
    env.reset(seed=7)
    tp = env.sim.object_state(env.target_name)[0]
    others = [env.sim.object_state(n)[0] for n in CUBES if n != env.target_name]
    for o in others:
        assert np.linalg.norm(o[:2] - tp[:2]) >= 0.06 - 1e-9
    assert np.linalg.norm(others[0][:2] - others[1][:2]) >= 0.06 - 1e-9
    env.set_stage('A')


def test_domain_randomization(env):
    """B 阶段物理参数在随机范围内；切回 A 恢复 MJCF 标称值。"""
    env.set_stage('B')
    env.reset(seed=11)
    m, bid = env.sim.model, env.sim.model.body(env.target_name).id
    assert 0.018 <= m.body_mass[bid] <= 0.035
    assert 0.9 <= m.geom_friction[m.body_geomadr[bid]][0] <= 1.4
    env.set_stage('A')
    env.reset(seed=12)
    bid = env.sim.model.body(env.target_name).id
    assert m.body_mass[bid] == pytest.approx(0.025)
    assert m.geom_friction[m.body_geomadr[bid]][0] == pytest.approx(1.2)


def test_scripted_sequence_smoke(env):
    """M1-4 预演：直线下降 → 闭合 → 直线抬升完整跑通，奖励结构合理。"""
    env.set_stage('A')
    env.reset(seed=5)
    total, status = 0.0, None
    for phase in (np.array([0, 0, -1, 0, 1], dtype=np.float32),   # 下降（夹爪全开）
                  np.array([0, 0, 0, 0, -1], dtype=np.float32),   # 闭合 1.2 s
                  np.array([0, 0, 1, 0, -1], dtype=np.float32)):  # 抬升
        for _ in range(40):
            _, r, term, trunc, info = env.step(phase)
            total += r
            if term or trunc:
                status = info['status']
                break
        if status is not None:
            break
    assert np.isfinite(total) and total > 0
    assert status in (None, 'success', 'dropped', 'timeout',
                      'table_contact', 'out_of_bounds')


# —— 方案 A：PBRS 奖励性质测试（无扰动口径，与 smoke_check 一致） ————————————

@pytest.fixture(scope='module')
def renv():
    """奖励测试专用环境：阶段 A、无感知噪声、无物理随机化。"""
    e = Fr3GraspEnv(stage='A')
    e.stage_params['noise_scale'] = 0.0
    e.stage_params['domain_rand'] = False
    yield e
    e.sim.close()


def _run_fixed(env, action_fn, seed):
    """以固定动作策略跑完一个回合，返回 (总奖励, 结局, 步数)。"""
    env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(action_fn(env))
        total += r
        done = term or trunc
    return total, info['status'], env.ctx.step


def test_shaping_telescopes(renv):
    """γ=1 时塑形项精确 telescoping（PBRS 核心性质）：整条轨迹
    Σ shape = Φ(s_T') − Φ(s₀)，其中终止步 Φ' = 0、截断步照常取 Φ'。
    与路径无关——用上飞（必终止）和随机动作（可能超时）两条路径验证。"""
    renv.reward_gamma = 1.0
    try:
        # ① 上飞（必然 out_of_bounds 终止）：Σ shape = −Φ(s₀)
        renv.reset(seed=100)
        phi0 = renv.phi_prev
        fly = np.array([0, 0, 1, 0, 1], dtype=np.float32)
        total, done, term = 0.0, False, False
        while not done:
            _, _, term, trunc, info = renv.step(fly)
            total += info['reward_terms']['shape']
            done = term or trunc
        assert term and info['status'] == 'out_of_bounds'
        assert total == pytest.approx(-phi0, abs=1e-6)
        # ② 随机动作（可能 timeout 截断）：Σ shape = Φ(s_T) − Φ(s₀)
        rng = np.random.default_rng(9)
        for k in range(3):
            renv.reset(seed=200 + k)
            phi0 = renv.phi_prev
            total, done = 0.0, False
            while not done:
                _, r, term, trunc, info = renv.step(rng.uniform(-1, 1, 5))
                total += info['reward_terms']['shape']
                done = term or trunc
            assert total == pytest.approx(info['reward_terms']['phi'] - phi0, abs=1e-6)
    finally:
        renv.reward_gamma = float(renv.conf['reward']['gamma'])


def test_fly_up_not_profitable(renv):
    """上飞自杀（Run3 学到的漏洞）：out_of_bounds 终止，总奖励 < −4，
    且低于原地悬停到超时的总奖励。"""
    fly = np.array([0, 0, 1, 0, 1], dtype=np.float32)
    totals = [_run_fixed(renv, lambda e: fly, 300 + i) for i in range(3)]
    assert all(s == 'out_of_bounds' for _, s, _ in totals)
    assert max(t for t, _, _ in totals) < -4.0
    hover_total, hover_status, _ = _run_fixed(renv, lambda e: np.zeros(5, dtype=np.float32), 400)
    assert hover_status == 'timeout'
    assert max(t for t, _, _ in totals) < hover_total


def test_hover_not_profitable(renv):
    """原地悬停（Run1 学到的漏洞）到超时：总奖励 ≤ 0.5。"""
    total, status, steps = _run_fixed(renv, lambda e: np.zeros(5, dtype=np.float32), 500)
    assert status == 'timeout' and steps == renv.cfg.max_steps
    assert total <= 0.5


def test_scripted_dominates(renv):
    """脚本基线成功回合总奖励 > 15（success ≫ 漏洞策略，结局排序正确）。"""
    hold = np.array([0, 0, 0, 0, -1], dtype=np.float32)
    for i in range(3):
        renv.reset(seed=600 + i)
        actions = list(scripted_actions(renv)) + [hold] * renv.cfg.max_steps
        total, status, done = 0.0, None, False
        for action in actions:
            _, r, term, trunc, info = renv.step(action)
            total += r
            done = term or trunc
            if done:
                status = info['status']
                break
        assert status == 'success', f'episode {i}: {status}'
        assert total > 15.0, f'episode {i}: total={total:.2f}'


def test_phi_range(renv):
    """任意状态下 0 ≤ Φ ≤ Φ_max = Σw（各分量都在 [0,1]）。
    覆盖脚本（可达高势）与随机（大范围游走）两条路径。"""
    w = renv.conf['reward']
    phi_max = (w['w_reach'] + w['w_yaw'] + w['w_close'] + w['w_grasp'] + w['w_lift'])
    rng = np.random.default_rng(11)
    # 脚本路径
    renv.reset(seed=700)
    hold = np.array([0, 0, 0, 0, -1], dtype=np.float32)
    for action in list(scripted_actions(renv)) + [hold] * 30:
        phi = renv._phi()
        assert 0.0 <= phi <= phi_max
        _, _, term, trunc, _ = renv.step(action)
        if term or trunc:
            break
    # 随机路径
    renv.reset(seed=701)
    for _ in range(80):
        phi = renv._phi()
        assert 0.0 <= phi <= phi_max
        _, _, term, trunc, _ = renv.step(rng.uniform(-1, 1, 5))
        if term or trunc:
            renv.reset(seed=702)
