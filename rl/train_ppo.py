"""PPO 训练脚本（方案 4.5 训练配置 + 4.4④ 课程晋升）。

课程 ladder：A → B → C(yaw 15°→30°→45°) → D(min_dist 0.10→0.06)。
晋升判据：滚动 window（默认 500）回合成功率 ≥ promote_at（默认 0.9），
在 rollout 边界检查，通过 env_method('configure') 广播到所有子环境。
观测归一化用 VecNormalize（norm_obs），导出 NumpyPolicy 时随 checkpoint
保存 obs_rms（M2-3 处理）。奖励不归一化（量级已按 4.3 表设计）。

正式训练（4090 机器）：
  python rl/train_ppo.py                       # 默认 12 env、30M 步
  tensorboard --logdir rl/runs                 # 监控
CPU 冒烟（本仓库容器）：
  python rl/train_ppo.py --n-envs 4 --total-timesteps 4096 --run-dir rl/runs/smoke
  python rl/train_ppo.py --n-envs 4 --total-timesteps 4096 --window 10 --promote-at 0 \
      --run-dir rl/runs/smoke_promote    # 验证晋升机制立即触发
E1 短训（§5 兜底；容器内防 memcg OOM：n-envs ≤ 8、OMP 4、-u 防日志缓冲丢失）：
  OMP_NUM_THREADS=4 python -u rl/train_ppo.py --n-envs 8 \
      --total-timesteps 3000000 --lr-schedule constant --run-dir rl/runs/e1_a0
E2 兜底（探索不到时；BC 预热 actor+critic → PPO 微调。v4：BC 在 A 档蒸馏，
--start-ladder 1 跳过 A0 消除分布跳变，--lr 收紧防首个更新推飞策略）：
  python rl/bc_pretrain.py --stage A --episodes 200
  OMP_NUM_THREADS=4 python -u rl/train_ppo.py --n-envs 8 \
      --init-bc rl/runs/bc_pretrain --start-ladder 1 --lr 5e-5 \
      --total-timesteps 3000000 --lr-schedule constant --run-dir rl/runs/e2_bc_v4
"""
import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'franka_ros2_ws/src/fr3_vision_grasp'))

from fr3_grasp_env import Fr3GraspEnv

# 课程阶梯：yaml 存各阶段最终值，C/D 的渐进细分由这里覆盖（4.4④）。
# A0 = E1 探索兜底（rl_reward_fix_plan.md §5）：6 cm 交接，晋升后回到 A（12 cm）。
LADDER = [
    ('A0', {}),
    ('A', {}),
    ('B', {}),
    ('C', {'yaw_deg': 15}), ('C', {'yaw_deg': 30}), ('C', {'yaw_deg': 45}),
    ('D', {'min_dist': 0.10}), ('D', {'min_dist': 0.06}),
]

# PPO 超参（方案 4.5：参考脚本迁移 + SB3 默认值）
LR = 3e-4
HYPERS = dict(n_steps=256, batch_size=1024, ent_coef=0.005, target_kl=0.03,
              gamma=0.99, gae_lambda=0.95, clip_range=0.2, n_epochs=10,
              net_arch=[512, 256, 128])


class CurriculumCallback(BaseCallback):
    """课程晋升：滚动 window 回合成功率 ≥ promote_at → 沿 ladder 升一级。

    is_success 与结局 status 只在回合结束（done=True）的 info 里收集（A1 修复：
    旧版对每步 info 无 done 门控地 append，即使成功率 100%，窗口均值上限也
    只有 ≈1/回合长度，永远无法晋升）。episode/* 记录每个 rollout 内各结局的
    占比；晋升只在 rollout 边界执行并清空窗口（新档位重新累积证据）。
    状态 dump 到 run_dir/curriculum.json 供续训参考。
    """

    def __init__(self, ladder, window=500, promote_at=0.9, run_dir=None, verbose=1):
        super().__init__(verbose)
        self.ladder = ladder
        self.window_size = window
        self.promote_at = promote_at
        self.run_dir = Path(run_dir) if run_dir else None
        self.window = deque(maxlen=window)
        self.status_counts = Counter()      # rollout 内各结局的回合数
        self.idx = 0

    def _collect(self, infos, dones):
        """回合结束才计数：成功窗口 + 结局统计（独立方法方便单元测试）。"""
        for info, done in zip(infos, dones):
            if done:
                self.window.append(bool(info.get('is_success', False)))
                self.status_counts[info.get('status') or 'unknown'] += 1

    def _on_training_start(self):
        stage, ov = self.ladder[0]
        self.training_env.env_method('configure', stage, **ov)
        self.level_start = self.num_timesteps
        self._dump()
        if self.verbose:
            print(f'[curriculum] start at ladder 0: stage {stage} {ov or ""}', flush=True)

    def _on_step(self):
        self._collect(self.locals['infos'], self.locals['dones'])
        if self.n_calls % self.model.n_steps:
            return True                     # 只在 rollout 边界检查
        # 各结局占比：单一结局占 100% = 出现漏洞策略，第一天就能看出来
        total = sum(self.status_counts.values()) or 1
        for k in ('success', 'timeout', 'out_of_bounds', 'table_contact', 'dropped'):
            self.logger.record(f'episode/{k}', self.status_counts[k] / total)
        self.status_counts.clear()
        rate = float(np.mean(self.window)) if self.window else 0.0
        self.logger.record('curriculum/success_rate', rate)
        self.logger.record('curriculum/ladder_idx', self.idx)
        self.logger.record('curriculum/level_steps', self.num_timesteps - self.level_start)
        if len(self.window) >= self.window_size and rate >= self.promote_at \
                and self.idx < len(self.ladder) - 1:
            self.idx += 1
            stage, ov = self.ladder[self.idx]
            self.training_env.env_method('configure', stage, **ov)
            self.window.clear()
            self.level_start = self.num_timesteps
            if self.verbose:
                print(f'\n[curriculum] ladder → {self.idx}: stage {stage} {ov or ""} '
                      f'(success {rate:.0%} over {self.window_size} episodes, '
                      f'step {self.num_timesteps})', flush=True)
            self._dump()
        return True

    def _dump(self):
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / 'curriculum.json').write_text(json.dumps(
                {'ladder_idx': self.idx, 'stage': self.ladder[self.idx][0],
                 'overrides': self.ladder[self.idx][1],
                 'level_start_step': int(self.level_start)}, ensure_ascii=False))


def make_env(seed):
    def _init():
        return Monitor(Fr3GraspEnv())       # 初始档由 CurriculumCallback 统一下发
    return _init


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n-envs', type=int, default=12)
    parser.add_argument('--total-timesteps', type=int, default=30_000_000)
    parser.add_argument('--start-ladder', type=int, default=0,
                        help='起始档位索引（0-7，断点续训用）')
    parser.add_argument('--lr-schedule', default='linear',
                        choices=['linear', 'constant'],
                        help='linear=随总步数衰减到 0（正式训练）；constant=常数'
                             '（短训用，防 LR 提前衰减冻结学习）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率覆盖。默认 3e-4；--init-bc 时自动减半 1.5e-4。'
                             'BC 微调建议 5e-5（E2 v3 教训：大 lr + 未蒸馏 critic '
                             '的首个更新 approx_kl=1.01，策略直接被推飞）')
    parser.add_argument('--window', type=int, default=500, help='晋升判据的回合窗口')
    parser.add_argument('--promote-at', type=float, default=0.9, help='晋升成功率阈值')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--resume', default=None, help='checkpoint .zip 路径（续训）')
    parser.add_argument('--init-bc', default=None, metavar='DIR',
                        help='E2：BC 预热目录（bc_pretrain.py 产出，含 '
                             'bc_pretrained.zip + vecnormalize.pkl）；lr 自动减半')
    parser.add_argument('--vecnormalize', default=None,
                        help='VecNormalize .pkl（默认自动探测 checkpoint 同名 _vecnormalize.pkl）')
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else HERE / 'runs' / time.strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'run dir: {run_dir}')

    # PBRS 一致性校验（方案 A1）：奖励塑形 r = γ·Φ(s') − Φ(s) 中的 γ 必须与
    # PPO 的 gamma 相同，telescoping 性质才成立；不一致直接报错退出
    with open(HERE / 'configs/grasp.yaml') as f:
        reward_gamma = float(yaml.safe_load(f)['reward']['gamma'])
    if reward_gamma != HYPERS['gamma']:
        sys.exit(f'configs/grasp.yaml 的 reward.gamma={reward_gamma} 与 PPO '
                 f'gamma={HYPERS["gamma"]} 不一致（PBRS 要求两者相同）')

    raw = SubprocVecEnv([make_env(args.seed) for _ in range(args.n_envs)])
    if args.resume:
        # 恢复伴随的 VecNormalize 统计：旧模型是在旧统计下训练的，
        # 若全新初始化会导致续训初期观测分布错乱
        from eval_policy import find_vecnormalize
        vec_path = Path(args.vecnormalize) if args.vecnormalize else find_vecnormalize(args.resume)
        venv = VecNormalize.load(str(vec_path), raw)
        print(f'resumed vecnormalize stats from {vec_path}', flush=True)
    elif args.init_bc:
        # E2：BC 在 raw obs 上蒸馏（bc_pretrain.py 的口径），PPO 沿用恒等
        # 归一化。归一化曾两次出问题：统计空/统计突变 + 低方差维度
        # （A0 档 cos(dyaw)）被放大千倍导致带噪 rollout 崩坏
        venv = VecNormalize(raw, norm_obs=False, norm_reward=False,
                            gamma=HYPERS['gamma'])
        print('init-bc: obs normalization disabled (BC raw-obs 口径)', flush=True)
    else:
        venv = VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            gamma=HYPERS['gamma'])
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' \
        else args.device
    print(f'device: {device} (cuda available: {torch.cuda.is_available()})', flush=True)

    # E2（§5）：BC 预热加载时 lr 减半，防 PPO 初期大步更新把蒸馏结果冲掉；
    # --lr 显式覆盖优先（BC 微调建议 5e-5）
    if args.lr is not None:
        base_lr = args.lr
    else:
        base_lr = LR * 0.5 if args.init_bc else LR
    lr = base_lr if args.lr_schedule == 'constant' else (lambda p: base_lr * p)
    policy_kwargs = dict(net_arch=HYPERS['net_arch'], activation_fn=torch.nn.Tanh)
    common = dict(tensorboard_log=str(run_dir), verbose=1, seed=args.seed, device=device,
                  ent_coef=HYPERS['ent_coef'], target_kl=HYPERS['target_kl'],
                  gamma=HYPERS['gamma'], gae_lambda=HYPERS['gae_lambda'],
                  clip_range=HYPERS['clip_range'], n_epochs=HYPERS['n_epochs'])
    if args.resume:
        model = PPO.load(args.resume, env=venv, learning_rate=lr, **common)
        print(f'resumed from {args.resume}, starting at ladder {args.start_ladder}',
              flush=True)
    elif args.init_bc:
        model = PPO.load(str(Path(args.init_bc) / 'bc_pretrained.zip'), env=venv,
                         learning_rate=lr, **common)
        print(f'BC-initialized from {args.init_bc} (lr halved to {base_lr})', flush=True)
    else:
        model = PPO('MlpPolicy', venv, n_steps=HYPERS['n_steps'],
                    batch_size=HYPERS['batch_size'],
                    learning_rate=lr,               # linear 时线性衰减至 0
                    policy_kwargs=policy_kwargs, **common)

    curriculum = CurriculumCallback(LADDER[args.start_ladder:], window=args.window,
                                    promote_at=args.promote_at, run_dir=run_dir)
    ckpt = CheckpointCallback(save_freq=HYPERS['n_steps'] * 50,   # 每 50 rollout
                              save_path=str(run_dir / 'checkpoints'), name_prefix='ppo',
                              save_vecnormalize=True)
    model.learn(total_timesteps=args.total_timesteps,
                callback=[curriculum, ckpt], progress_bar=False)
    final = run_dir / 'checkpoints' / 'ppo_final'
    model.save(final)
    venv.save(str(run_dir / 'checkpoints' / 'vecnormalize_final.pkl'))
    print(f'saved final model to {final}.zip')


if __name__ == '__main__':
    main()
