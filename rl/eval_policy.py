"""评估与导出（M2-3）：RL checkpoint 各课程档成功率 + NumpyPolicy 导出。

三种模式（评估器自校准，无扰动口径与 scripted_baseline 完全一致）：
  --mode scripted   脚本基线序列（自检：应复现 100%）
  --mode random     随机策略（自检：成功率 ≈ 0）
  --mode rl         加载 PPO checkpoint 评估（--model xxx.zip）
无扰动 = noise_scale 0 + domain_rand 关闭（--noisy 可保留）。
--export out.npz：导出 NumpyPolicy 格式权重（policy_net + action_net +
VecNormalize obs 统计），导出时自动做前向一致性检查（vs SB3 predict）。

运行示例：
  python rl/eval_policy.py --mode scripted                      # 评估器自检
  python rl/eval_policy.py --mode rl --model rl/runs/<run>/checkpoints/ppo_final.zip \
      --export rl/runs/<run>/policy.npz                         # 正式评估 + 导出
  python rl/eval_policy.py --mode rl --model rl/runs/<run>/checkpoints/ppo_final.zip \
      --episodes 20 --diagnose                                  # 行为诊断：下降/闭合/终态细节
"""
import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'franka_ros2_ws/src/fr3_vision_grasp'))

from fr3_grasp_env import Fr3GraspEnv
from fr3_vision_grasp.grasp_policy import NumpyPolicy
from scripted_baseline import scripted_actions

HOLD = np.array([0, 0, 0, 0, -1], dtype=np.float32)


def load_vec_stats(path):
    """从 VecNormalize pickle 读 obs 归一化统计（obs_rms / clip_obs）。

    E2 v4 链路 norm_obs=False：SB3 新版此时不创建 obs_rms 属性（且裸
    pickle.load 后访问缺失属性会触发 __getattr__ 递归崩溃），此时返回
    恒等统计 mean=0/var=1/clip=∞——normalize() 退化为直通，与训练口径
    （VecNormalize(norm_obs=False) 下 policy 输入即 raw obs）一致。
    """
    with open(path, 'rb') as f:
        vn = pickle.load(f)
    d = vars(vn)
    if not d.get('norm_obs') or 'obs_rms' not in d:
        return np.zeros(1), np.ones(1), 1e9
    return d['obs_rms'].mean.copy(), d['obs_rms'].var.copy(), float(d['clip_obs'])


def find_vecnormalize(model_path):
    """按命名约定探测 checkpoint 伴随的 VecNormalize 统计文件。

    CheckpointCallback：ppo_<n>_steps.zip ↔ ppo_vecnormalize_<n>_steps.pkl；
    train_ppo.py 的 final 保存：ppo_final.zip ↔ vecnormalize_final.pkl。
    """
    import re
    p = Path(model_path)
    cands = []
    m = re.match(r'(.+)_(\d+)_steps$', p.stem)
    if m:
        prefix, num = m.groups()
        cands.append(p.with_name(f'{prefix}_vecnormalize_{num}_steps.pkl'))
    cands.append(p.parent / 'vecnormalize_final.pkl')
    cands.append(p.with_name(p.stem + '_vecnormalize.pkl'))
    for cand in cands:
        if cand.exists():
            return cand
    raise FileNotFoundError(f'未找到 VecNormalize 统计（尝试过 {", ".join(str(c) for c in cands)}），'
                            f'请用 --vecnormalize 指定')


def normalize(obs, mean, var, clip):
    return np.clip((obs - mean) / np.sqrt(var + 1e-8), -clip, clip)


def export_npz(model, vec_path, out_path, n_check=64):
    """提取 policy_net/action_net 权重 + obs 统计 → npz，并做前向一致性检查。

    一致性口径：随机原始 obs 上，SB3 predict(normalize(obs)) 与
    NumpyPolicy.act(obs)（内部自行归一化）都在 clip[-1,1] 后比较。
    """
    import torch
    weights, biases = [], []
    for layer in model.policy.mlp_extractor.policy_net:
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy())
            biases.append(layer.bias.detach().cpu().numpy())
    mean, var, clip = load_vec_stats(vec_path)
    arrays = {f'pi_w{i}': w for i, w in enumerate(weights)}
    arrays.update({f'pi_b{i}': b for i, b in enumerate(biases)})
    arrays.update(action_w=model.policy.action_net.weight.detach().cpu().numpy(),
                  action_b=model.policy.action_net.bias.detach().cpu().numpy(),
                  obs_mean=mean, obs_var=var, clip_obs=np.float64(clip))
    np.savez(out_path, **arrays)

    policy = NumpyPolicy(out_path)
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(n_check):
        raw = rng.normal(0.0, 1.0, model.observation_space.shape)
        a_sb, _ = model.predict(normalize(raw, mean, var, clip), deterministic=True)
        max_err = max(max_err, float(np.max(np.abs(
            np.clip(a_sb, -1, 1) - policy.act(raw)))))
    status = 'PASS' if max_err < 1e-5 else 'FAIL'
    print(f'[export] {out_path}  一致性检查 {status}  max|Δ|={max_err:.2e} '
          f'({len(weights)} 隐藏层, clip_obs={clip})')
    if max_err >= 1e-5:
        raise AssertionError('NumpyPolicy 前向与 SB3 不一致，禁止部署该导出')


def make_action_fn(mode, env, model=None, stats=None, rng=None):
    """返回 action_fn(obs)：按模式给出动作。scripted 用 ctx.step==0 检测回合开始。"""
    if mode == 'scripted':
        def action_fn(obs):
            if env.ctx.step == 0:
                seq = list(scripted_actions(env)) + [HOLD] * (env.cfg.max_steps + 10)
                env._eval_seq = iter(seq)
            return next(env._eval_seq)
        return action_fn
    if mode == 'random':
        return lambda obs: rng.uniform(-1.0, 1.0, env.cfg.action_dim)
    mean, var, clip = stats

    def action_fn(obs):
        a, _ = model.predict(normalize(obs, mean, var, clip), deterministic=True)
        return a
    return action_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mode', default='rl', choices=['rl', 'scripted', 'random'])
    parser.add_argument('--model', help='PPO checkpoint .zip（--mode rl 必填）')
    parser.add_argument('--vecnormalize', default=None, help='VecNormalize .pkl（默认自动探测）')
    parser.add_argument('--export', default=None, help='导出 NumpyPolicy npz 路径')
    parser.add_argument('--stage', default='A', choices=list('ABCD'))
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noisy', action='store_true', help='保留噪声与物理随机化')
    parser.add_argument('--fixed', action='store_true', help='方块固定 (0.45, 0)')
    parser.add_argument('--diagnose', action='store_true',
                        help='逐回合行为诊断：TCP 下降深度、到抓取点距离、'
                             '夹爪闭合（命令/实际指宽）、close 通道动作均值')
    args = parser.parse_args()

    model, stats, rng = None, None, None
    if args.mode == 'rl':
        if not args.model:
            parser.error('--mode rl 需要 --model')
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device='cpu')
        vec_path = Path(args.vecnormalize) if args.vecnormalize else find_vecnormalize(args.model)
        stats = load_vec_stats(vec_path)
        print(f'model: {args.model}\nvecnormalize: {vec_path}')
    elif args.export:
        parser.error('--export 仅在 --mode rl 下有意义')
    rng = np.random.default_rng(args.seed + 7777)

    env = Fr3GraspEnv(stage=args.stage)
    if not args.noisy:      # 与 scripted_baseline 相同的无扰动口径
        env.stage_params['noise_scale'] = 0.0
        env.stage_params['domain_rand'] = False
    fixed = np.array([0.45, 0.0]) if args.fixed else None
    action_fn = make_action_fn(args.mode, env, model, stats, rng)

    counts, steps, diag = Counter(), [], []
    gap = env.conf['reward']['grasp_gap']
    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + i,
                           options={'target_xy': fixed} if fixed is not None else None)
        rec = {'z0': float(env.sim.data.site_xpos[env.ctx.tcp][2]), 'zmin': np.inf,
               'd_end': np.inf, 'cmd_min': np.inf, 'w_min': np.inf, 'a4': [], 'r': 0.0}
        done = False
        while not done:
            act = action_fn(obs)
            obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            if args.diagnose:
                rec['r'] += rew
                rec['a4'].append(float(act[4]))
                rec['zmin'] = min(rec['zmin'], float(env.sim.data.site_xpos[env.ctx.tcp][2]))
                cube = env.sim.object_state(env.target_name)[0]
                rec['d_end'] = float(np.linalg.norm(
                    env.sim.data.site_xpos[env.ctx.tcp]
                    - (cube + np.array([0.0, 0.0, gap]))))
                rec['cmd_min'] = min(rec['cmd_min'],
                                     2.0 * float(env.sim.data.ctrl[env.ctx.gripper]))
                rec['w_min'] = min(rec['w_min'],
                                   float(env.sim.data.qpos[env.ctx.fqadr].sum()))
        counts[info['status']] += 1
        steps.append(env.ctx.step)
        rec['status'], rec['steps'] = info['status'], env.ctx.step
        diag.append(rec)
    env.sim.close()

    n = args.episodes
    print(f'mode={args.mode} stage={args.stage} fixed={fixed is not None} '
          f'noisy={args.noisy} episodes={n}')
    for key in ('success', 'dropped', 'table_contact', 'out_of_bounds', 'timeout', 'no_end'):
        if counts[key]:
            print(f'  {key:14s} {counts[key]:4d}  ({counts[key] / n:.0%})')
    rate = counts['success'] / n
    print(f'  success rate: {rate:.0%}, avg steps: {np.mean(steps):.1f}  '
          f'(脚本基线 M1-4: 100%)')

    if args.diagnose:
        descend = sum(1 for r in diag if r['zmin'] < r['z0'] - 0.03)
        arrive = sum(1 for r in diag if r['d_end'] < 0.02)
        cmd_close = sum(1 for r in diag if r['cmd_min'] < 0.04)
        real_close = sum(1 for r in diag if r['w_min'] < 0.04)
        print(f'\n行为诊断（{n} 回合，确定性动作，无扰动口径）:')
        print(f'  下降超 3cm（zmin < z0-0.03）  : {descend}/{n}')
        print(f'  到达抓取点 2cm 内（d_end）     : {arrive}/{n}')
        print(f'  命令闭合过半（cmd宽 < 40mm）   : {cmd_close}/{n}')
        print(f'  实际指宽 < 40mm                : {real_close}/{n}')
        print(f'  z0 均值 {np.mean([r["z0"] for r in diag]):.3f} | '
              f'zmin 均值 {np.mean([r["zmin"] for r in diag]):.3f} | '
              f'd_end 均值 {np.mean([r["d_end"] for r in diag]):.3f} | '
              f'cmd_min 均值 {np.mean([r["cmd_min"] for r in diag]):.3f} | '
              f'w_min 均值 {np.mean([r["w_min"] for r in diag]):.3f}')
        print(f'  close 通道动作均值 {np.mean([np.mean(r["a4"]) for r in diag]):+.3f} | '
              f'回合总奖励均值 {np.mean([r["r"] for r in diag]):+.2f}')
        tail = ' | '.join(f'{r["status"][:4]}:{r["steps"]}步 z{r["zmin"]:.3f} d{r["d_end"]:.3f}'
                          for r in diag[:10])
        print(f'  逐回合: {tail}' + (' ...' if n > 10 else ''))

    if args.export:
        export_npz(model, vec_path, args.export)


if __name__ == '__main__':
    main()
