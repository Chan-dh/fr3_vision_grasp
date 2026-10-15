"""脚本基线：复现 pick.run_pick 抓取段的策略（方案 M1-4 / M1 通过标准②）。

直线下降到感知方块中心高度（run_pick 的 move_to(center, linear=True)，
TCP ≈ 0.065，z_min=0.063 之上）→ 全闭（hand(0)，1 s）→ 直线抬升
lift_height + 余量。在 Fr3GraspEnv 中统计无扰动成功率：
  1. 与 ROS 端实测对照（同一 MJCF、同一 ctrl 语义、同一物理步进）；
  2. 作为 M2 的对比基准（RL 成功率 ≥ 脚本基线）。
运行：source .venv-fr3/bin/activate && python rl/scripted_baseline.py --episodes 100
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'franka_ros2_ws/src/fr3_vision_grasp'))

from fr3_grasp_env import Fr3GraspEnv

CLOSE_STEPS = 20        # 闭合 1.0 s（伺服时间常数 ≈ 0.1 s，指宽充分稳定）


def scripted_actions(env):
    """按 run_pick 的节奏生成动作序列（下降全开 → 闭合 → 抬升保持闭合）。"""
    cfg = env.cfg
    perceived = env.ctx.perceived
    n_down = int(np.ceil((env.controller.cmd_pos[2] - perceived[2]) / cfg.max_dpos))
    n_up = int(np.ceil((cfg.lift_height + 0.02) / cfg.max_dpos))   # 余量补伺服滞后
    down = np.array([0, 0, -1, 0, 1], dtype=np.float32)
    close = np.array([0, 0, 0, 0, -1], dtype=np.float32)
    up = np.array([0, 0, 1, 0, -1], dtype=np.float32)
    return [down] * n_down + [close] * CLOSE_STEPS + [up] * n_up


def run_episode(env, seed, fixed_xy=None):
    """跑一个回合，返回最终 status。序列结束后仍保持闭合悬停到回合自然结束。"""
    options = {'target_xy': fixed_xy} if fixed_xy is not None else None
    env.reset(seed=seed, options=options)
    hold = np.array([0, 0, 0, 0, -1], dtype=np.float32)
    actions = scripted_actions(env) + [hold] * env.cfg.max_steps
    for action in actions:
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            return info['status']
    return 'no_end'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--stage', default='A', choices=['A0'] + list('ABCD'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fixed', action='store_true',
                        help='方块固定在 (0.45, 0)（ROS 场景红方块位置）')
    parser.add_argument('--noisy', action='store_true',
                        help='保留感知噪声与物理随机化（默认关闭，做无扰动统计）')
    args = parser.parse_args()

    env = Fr3GraspEnv(stage=args.stage)
    if not args.noisy:
        env.stage_params['noise_scale'] = 0.0
        env.stage_params['domain_rand'] = False
    fixed = np.array([0.45, 0.0]) if args.fixed else None
    counts, steps = Counter(), []
    for i in range(args.episodes):
        status = run_episode(env, args.seed + i, fixed)
        counts[status] += 1
        steps.append(env.ctx.step)
    env.sim.close()

    n = args.episodes
    print(f'stage={args.stage} fixed={fixed is not None} '
          f'noisy={args.noisy} episodes={n}')
    for key in ('success', 'dropped', 'table_contact', 'out_of_bounds', 'timeout', 'no_end'):
        if counts[key]:
            print(f'  {key:14s} {counts[key]:4d}  ({counts[key] / n:.0%})')
    print(f'  success rate: {counts["success"] / n:.0%}, avg steps: {np.mean(steps):.1f}')


if __name__ == '__main__':
    main()
