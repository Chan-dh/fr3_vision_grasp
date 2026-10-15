"""M2-1 冒烟验证：训练前奖励与终止信号的方向性检查（不训练）。

方案 A（docs/rl_reward_fix_plan.md A5）：在脚本、随机策略之外，加入三种
漏洞策略对照（上飞自杀=Run3、原地悬停=Run1、下降到抓取点悬停不闭合），
验证 PBRS 奖励下所有漏洞策略都不划算。各策略跑 N 回合（默认阶段 A 无
扰动，与 M1-4 基线同口径），检查项：
  1. 全程 reward / obs 无 NaN；
  2. 脚本策略成功率 ≈ 100%，随机策略 ≈ 0%（终止信号方向正确）；
  3. 脚本平均总奖励显著高于其他策略（奖励信号方向正确）；
  4. 单步奖励在合理范围内（无爆炸/异常尖峰）；
  5. success 回合的总奖励明显高于其他结局（判据与奖励一致）；
  6. 三种漏洞策略平均总奖励都 ≤ 1.0（赚不到塑形分）；
  7. 上飞 < 悬停（越早自杀越亏）；
  8. 脚本 − max(漏洞策略) ≥ 12（成功显著占优）。
全部通过打印 PASS 并退出码 0；任一失败打印 FAIL 并退出码 1。
运行：source .venv-fr3/bin/activate && python rl/smoke_check.py
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'franka_ros2_ws/src/fr3_vision_grasp'))

from fr3_grasp_env import Fr3GraspEnv
from scripted_baseline import scripted_actions

# 判定阈值（阶段 A 无扰动口径下的宽松界，超界说明奖励/终止实现有问题）
MAX_STEP_REWARD = 25.0    # 单步 |r| 上限（success 步 ≈ 20 − Φ ≈ 16；失败步 ≈ −5 − Φ）
SCRIPT_MIN_RATE = 0.95    # 脚本策略成功率下限（M1-4 实测 100%）
RANDOM_MAX_RATE = 0.05    # 随机策略成功率上限（期望 0）
REWARD_GAP = 12.0         # 脚本平均总奖励须高出其他策略的最少裕量
EXPLOIT_MAX_REWARD = 1.0  # 漏洞策略平均总奖励上限（悬停/上飞/下降悬停都赚不到分）
HOLD = np.array([0, 0, 0, 0, -1], dtype=np.float32)
FLY = np.array([0, 0, 1, 0, 1], dtype=np.float32)      # 上飞自杀（Run3 漏洞）
HOVER = np.zeros(5, dtype=np.float32)                   # 原地悬停（Run1 漏洞）


def scripted_policy(env, rng):
    """回合开始时按 baseline 逻辑生成动作序列，之后依次给出。"""
    if env.ctx.step == 0:
        seq = scripted_actions(env)
        env._smoke_seq = iter(list(seq) + [HOLD] * (env.cfg.max_steps + 10))
    return next(env._smoke_seq)


def descend_hover_policy(env, rng):
    """下降到抓取点后悬停、夹爪保持全开（不闭合）：能赚 reach/near 的势增量，
    但拿不到 grasp/lift 势——PBRS 下必须真的夹住提起才有分。"""
    if env.ctx.step == 0:
        cfg = env.cfg
        n_down = int(np.ceil((env.controller.cmd_pos[2] - env.ctx.perceived[2])
                             / cfg.max_dpos))
        down = np.array([0, 0, -1, 0, 1], dtype=np.float32)
        hold_open = np.array([0, 0, 0, 0, 1], dtype=np.float32)
        env._smoke_seq = iter([down] * n_down + [hold_open] * (cfg.max_steps + 10))
    return next(env._smoke_seq)


def fly_policy(env, rng):
    return FLY


def hover_policy(env, rng):
    return HOVER


def random_policy(env, rng):
    return rng.uniform(-1.0, 1.0, env.cfg.action_dim)


def run_policy(env, n_episodes, seed0, policy):
    """跑 N 回合，返回 (总奖励, 每回合 status, 步数, 最大单步 |r|, NaN 标记)。"""
    totals, statuses, steps = [], [], []
    rng = np.random.default_rng(seed0 + 7777)     # 策略 rng 与 env 噪声 rng 独立
    has_nan = False
    max_step = 0.0
    for i in range(n_episodes):
        env.reset(seed=seed0 + i)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy(env, rng))
            if not (np.isfinite(r) and np.all(np.isfinite(obs))):
                has_nan = True
            max_step = max(max_step, abs(r))
            total += r
            done = term or trunc
        totals.append(total)
        statuses.append(info['status'])
        steps.append(env.ctx.step)
    return totals, statuses, steps, max_step, has_nan


def summarize(name, totals, statuses, steps, max_step, has_nan):
    counts = Counter(statuses)
    n = len(totals)
    print(f'[{name}]')
    for key in ('success', 'dropped', 'table_contact', 'out_of_bounds', 'timeout', 'no_end'):
        if counts[key]:
            print(f'  {key:14s} {counts[key]:4d}  ({counts[key] / n:.0%})')
    print(f'  total reward: mean={np.mean(totals):7.1f}  std={np.std(totals):5.1f}  '
          f'min={np.min(totals):7.1f}  max={np.max(totals):7.1f}')
    print(f'  avg steps: {np.mean(steps):.1f}   max |step reward|: {max_step:.1f}   '
          f'NaN: {"有" if has_nan else "无"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--stage', default='A', choices=['A0'] + list('ABCD'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noisy', action='store_true', help='保留噪声与物理随机化')
    args = parser.parse_args()

    env = Fr3GraspEnv(stage=args.stage)
    if not args.noisy:      # 默认与 M1-4 基线同口径：无扰动
        env.stage_params['noise_scale'] = 0.0
        env.stage_params['domain_rand'] = False

    print(f'=== M2-1 冒烟验证：奖励与终止信号（stage={args.stage}, '
          f'episodes={args.episodes}, noisy={args.noisy}）===\n')
    policies = [('脚本策略', scripted_policy), ('随机策略', random_policy),
                ('上飞（漏洞）', fly_policy), ('悬停（漏洞）', hover_policy),
                ('下降悬停不闭合（漏洞）', descend_hover_policy)]
    results = {}
    for name, policy in policies:
        results[name] = run_policy(env, args.episodes, args.seed, policy)
        summarize(name, *results[name])
    env.sim.close()
    (s_tot, s_stat, s_steps, s_max, s_nan) = results['脚本策略']
    (r_tot, r_stat, r_steps, r_max, r_nan) = results['随机策略']
    (f_tot, f_stat, f_steps, f_max, f_nan) = results['上飞（漏洞）']
    (h_tot, h_stat, h_steps, h_max, h_nan) = results['悬停（漏洞）']
    (d_tot, d_stat, d_steps, d_max, d_nan) = results['下降悬停不闭合（漏洞）']

    # 按结局分组的平均总奖励（验证 success 与高奖励一致）
    print('\n按结局分组的平均总奖励：')
    for name, totals, statuses in (('脚本', s_tot, s_stat), ('随机', r_tot, r_stat)):
        groups = defaultdict(list)
        for total, status in zip(totals, statuses):
            groups[status].append(total)
        parts = ', '.join(f'{k} ({len(v)}): {np.mean(v):.1f}' for k, v in sorted(groups.items()))
        print(f'  {name}: {parts}')

    # —— 检查项 ——
    s_cnt, r_cnt = Counter(s_stat), Counter(r_stat)
    all_max_step = max(s_max, r_max, f_max, h_max, d_max)
    all_nan = any((s_nan, r_nan, f_nan, h_nan, d_nan))
    exploit_means = {'上飞': np.mean(f_tot), '悬停': np.mean(h_tot),
                     '下降悬停': np.mean(d_tot), '随机': np.mean(r_tot)}
    checks = []
    checks.append(('无 NaN（reward 与 obs）', not all_nan))
    s_rate = s_cnt['success'] / args.episodes
    r_rate = r_cnt['success'] / args.episodes
    checks.append((f'脚本成功率 {s_rate:.0%} ≥ {SCRIPT_MIN_RATE:.0%}',
                   s_rate >= SCRIPT_MIN_RATE))
    checks.append((f'随机成功率 {r_rate:.0%} ≤ {RANDOM_MAX_RATE:.0%}',
                   r_rate <= RANDOM_MAX_RATE))
    checks.append((f'单步 |r| ≤ {MAX_STEP_REWARD}', all_max_step <= MAX_STEP_REWARD))
    # 漏洞策略都不划算（PBRS：塑形收益与路径无关）
    for k, v in exploit_means.items():
        checks.append((f'{k}平均总奖励 {v:.1f} ≤ {EXPLOIT_MAX_REWARD}',
                       v <= EXPLOIT_MAX_REWARD))
    checks.append((f'上飞 {np.mean(f_tot):.1f} < 悬停 {np.mean(h_tot):.1f}',
                   np.mean(f_tot) < np.mean(h_tot)))
    gap = np.mean(s_tot) - max(exploit_means.values())
    checks.append((f'脚本 − max(其他策略) gap={gap:.1f} ≥ {REWARD_GAP}', gap >= REWARD_GAP))
    # success 回合奖励一致性：脚本 success 回合的平均总奖励须高于其他结局
    s_groups = defaultdict(list)
    for total, status in zip(s_tot, s_stat):
        s_groups[status].append(total)
    other = [v for k, vals in s_groups.items() if k != 'success' for v in vals]
    consistent = (np.mean(s_groups['success']) > np.mean(other)) if other else True
    checks.append((f'脚本 success 平均 {np.mean(s_groups["success"]):.1f} > '
                   f'非 success 平均 {np.mean(other) if other else float("nan"):.1f}',
                   consistent))

    print('\n检查项：')
    ok = True
    for desc, passed in checks:
        print(f'  [{"PASS" if passed else "FAIL"}] {desc}')
        ok = ok and passed
    print(f'\n结果：{"全部通过 ✓" if ok else "存在失败项 ✗"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
