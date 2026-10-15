"""E2 兜底（docs/rl_reward_fix_plan.md §5）：脚本基线 BC 预热 actor + critic。

fixA_short 短训判定为"探索不到"（success=0、timeout 100%）：随机探索采样
不到"下降→闭合→抬升→保持"的完整序列。E2 用 scripted_baseline 的
(obs, action) 对以 MSE 蒸馏 actor 均值头（小初始 std，见 --log-std-init），
让 PPO 从"已经会抓"的起点微调，代替随机策略冷启动探索。

E2 v4：critic（value 头）必须同时蒸馏——v3 只蒸馏 actor，critic 随机初始化，
PPO 的 GAE 优势全为噪声，首个 rollout 更新 approx_kl=1.01（target_kl 的
33 倍）直接把策略推飞；A0 档容错高掩盖了退化（success 仍 100%），晋升
A 档后崩坏（success 36%→4%，timeout 87%）。蒸馏档位也改为 A（12 cm 交接，
与 ladder 主线一致），消除 A0→A 分布跳变。

产出（--run-dir）：
  bc_pretrained.zip   供 train_ppo.py --init-bc 加载（E2 链路不归一化 obs，
                      train_ppo --init-bc 自动用恒等归一化，口径一致）

运行（容器 CPU，分钟级）：
  source .venv-fr3/bin/activate
  python rl/bc_pretrain.py --stage A --episodes 200
接 PPO 微调（--start-ladder 1 跳过 A0，从 A 档开始；--lr 收紧）：
  OMP_NUM_THREADS=4 python -u rl/train_ppo.py --n-envs 8 \
      --init-bc rl/runs/bc_pretrain --start-ladder 1 --lr 5e-5 \
      --total-timesteps 3000000 --lr-schedule constant --run-dir rl/runs/e2_bc_v4
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'franka_ros2_ws/src/fr3_vision_grasp'))

from fr3_grasp_env import Fr3GraspEnv
from scripted_baseline import scripted_actions
from train_ppo import HYPERS

BC_LR = 1e-4


def collect(env, episodes, seed0, hold):
    """采集脚本回合 (raw obs, action, 折扣回报 G_t)。

    与 run_episode 同节奏：脚本序列结束后保持闭合悬停到回合自然结束
    （成功后保持段也是要学的行为——判据④要求连续保持）。
    G_t = Σ γ^k·r_{t+k}（γ 与 PPO 一致）按回合倒序累积，供 critic 回归。
    """
    gamma = HYPERS['gamma']
    obs_raw, acts, returns, statuses = [], [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        ep_obs, ep_acts, ep_rews = [], [], []
        for a in scripted_actions(env) + [hold] * env.cfg.max_steps:
            ep_obs.append(obs.copy())
            ep_acts.append(a)
            obs, r, term, trunc, info = env.step(a)
            ep_rews.append(r)
            if term or trunc:
                statuses.append(info['status'])
                break
        g, ep_ret = 0.0, [0.0] * len(ep_rews)
        for t in reversed(range(len(ep_rews))):
            g = ep_rews[t] + gamma * g
            ep_ret[t] = g
        obs_raw.extend(ep_obs)
        acts.extend(ep_acts)
        returns.extend(ep_ret)
    return (np.asarray(obs_raw, np.float64), np.asarray(acts, np.float32),
            np.asarray(returns, np.float64), statuses)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stage', default='A', choices=['A0'] + list('ABCD'))
    parser.add_argument('--episodes', type=int, default=200,
                        help='脚本回合数（§5 建议 200）')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-std-init', type=float, default=-2.5,
                        help='初始探索 std=e^x。std 必须小到带噪轨迹仍能成功，'
                             '否则 PPO 首个 rollout 全失败、蒸馏成果被冲掉；'
                             'log_std 可学习，后续 PPO 自行调整')
    parser.add_argument('--run-dir', default='rl/runs/bc_pretrain')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = Fr3GraspEnv(stage=args.stage)
    venv = DummyVecEnv([lambda: env])     # E2 链路不归一化 obs（见下）

    print(f'collecting {args.episodes} scripted episodes (stage {args.stage}) ...',
          flush=True)
    hold = np.array([0, 0, 0, 0, -1], dtype=np.float32)
    obs_raw, acts, returns, statuses = collect(env, args.episodes, args.seed, hold)
    print(f'  {len(obs_raw)} samples, status: {dict(Counter(statuses))}', flush=True)

    # E2 链路全程不做 obs 归一化（bc_pretrain 与 train_ppo --init-bc 口径一致）：
    # 1) 本环境 obs 均为小量纲相对量，归一化收益小；
    # 2) A0 档 yaw=0 时 cos(dyaw) 等维度方差 ~1e-6，归一化把微小扰动放大
    #    千倍，带噪 rollout 直接触发分布漂移、策略崩坏（实测 std=0.08 也全灭）；
    # 3) 归一化统计的保存/加载口径在此前调试中已两次踩坑。
    obs_n = obs_raw.astype(np.float32)
    ret_n = returns.astype(np.float32)

    # 与 train_ppo.py 相同的网络结构；log_std_init 见 --log-std-init 说明。
    # 固定 torch 种子：网络初始化可复现（否则每次跑 BC 收敛质量有随机性）
    torch.manual_seed(args.seed)
    model = PPO('MlpPolicy', venv,
                policy_kwargs=dict(net_arch=HYPERS['net_arch'],
                                   activation_fn=torch.nn.Tanh,
                                   log_std_init=args.log_std_init))
    opt = torch.optim.Adam(model.policy.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    n = len(obs_n)
    dev = next(model.policy.parameters()).device
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        a_losses, v_losses = [], []
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            obs_t = torch.as_tensor(obs_n[idx], device=dev)
            act_t = torch.as_tensor(acts[idx], device=dev)
            ret_t = torch.as_tensor(ret_n[idx], device=dev)
            dist = model.policy.get_distribution(obs_t)
            values = model.policy.predict_values(obs_t).flatten()
            # actor：蒸馏脚本动作；critic：回归折扣回报（0.5 = PPO 的 value_coef，
            # 与 PPO loss 中 value 项权重一致，微调时量级平滑衔接）
            a_loss = F.mse_loss(dist.distribution.mean, act_t)
            v_loss = F.mse_loss(values, ret_t)
            loss = a_loss + 0.5 * v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            a_losses.append(a_loss.item())
            v_losses.append(v_loss.item())
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'  epoch {epoch + 1}/{args.epochs}  '
                  f'actor mse {np.mean(a_losses):.4f}  '
                  f'value mse {np.mean(v_losses):.4f}', flush=True)

    # 确定性自检：区分"口径 bug"（确定性也失败）与"噪声容差不足"（仅带噪失败）
    det_ok, det_out = 0, []
    for ep in range(5):
        obs, _ = env.reset(seed=args.seed + 20000 + ep)
        for _ in range(env.cfg.max_steps):
            a = model.predict(obs[None], deterministic=True)[0][0]
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                det_ok += int(info['status'] == 'success')
                det_out.append(info['status'])
                break
    print(f'BC 确定性自检: {det_ok}/5  结局: {det_out}', flush=True)

    # 带噪采样自检：BC 产物必须"加采样噪声仍成功"（train_ppo 的 rollout 用
    # mean+std·noise 采样，E2 首跑教训——只做确定性评估挡不住这类废权重）
    ok, outcomes = 0, []
    for ep in range(5):
        obs, _ = env.reset(seed=args.seed + 10000 + ep)
        for _ in range(env.cfg.max_steps):
            with torch.no_grad():
                o = torch.as_tensor(obs[None], dtype=torch.float32)
                a = model.policy.get_distribution(o).sample().numpy()[0]
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                ok += int(info['status'] == 'success')
                outcomes.append(info['status'])
                break
    print(f'BC 带噪采样自检: {ok}/5  结局: {outcomes}', flush=True)
    if ok < 4:
        sys.exit('自检未过（<4/5 成功）：不保存。建议加大 --epochs 或检查采集数据')

    model.save(str(run_dir / 'bc_pretrained.zip'))
    print(f'saved: {run_dir / "bc_pretrained.zip"}')
    env.sim.close()


if __name__ == '__main__':
    main()
