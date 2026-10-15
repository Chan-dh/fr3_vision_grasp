"""训练结果可视化回放：PPO 策略在 Fr3GraspEnv 中的演示。

两种模式（二选一，--video 优先）：
  --video out.mp4   离屏渲染视频（EGL 后端，headless 容器可用，已验证）；
                    每控制步 1 帧、20 fps ≈ 实时速率
  --viewer          交互式 viewer（mujoco.viewer.launch_passive，需要显示
                    环境：本地桌面或 ssh -X 转发；容器内无显示会报错）

obs 口径自动识别：checkpoint 伴随的 VecNormalize pkl 若 norm_obs=False
（E2 v4 链路：raw obs 训练）则恒等直通；若未来跑归一化训练则自动套用统计。

运行：
  python rl/play_policy.py --model rl/runs/e2_bc_v4/checkpoints/ppo_final.zip \
      --stage D --video rl/runs/e2_bc_v4/play.mp4
  python rl/play_policy.py --model ... --viewer --episodes 5    # 本地机器
"""
import argparse
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'franka_ros2_ws/src/fr3_vision_grasp'))

from eval_policy import find_vecnormalize


def make_obs_adapter(model_path):
    """按 checkpoint 伴随 pkl 的 norm_obs 标志返回 obs→policy输入 的适配器。"""
    try:
        p = find_vecnormalize(model_path)
    except FileNotFoundError:
        return lambda o: o, 'raw（未找到 vecnormalize pkl，假定不归一化）'
    with open(p, 'rb') as f:
        vn = pickle.load(f)
    # vars() 访问：SB3 新版 norm_obs=False 时不创建 obs_rms 属性，
    # 裸属性访问会触发 __getattr__ 递归崩溃
    d = vars(vn)
    if not d.get('norm_obs') or 'obs_rms' not in d:
        return lambda o: o, 'raw（norm_obs=False，恒等直通）'
    mean, var, clip = d['obs_rms'].mean, d['obs_rms'].var, float(d['clip_obs'])

    def adapt(o):
        return np.clip((o - mean) / np.sqrt(var + 1e-8), -clip, clip)
    return adapt, f'normalized（mean/var 来自 {p.name}）'


def run_episodes(env, model, adapt, episodes, seed, noisy, on_step):
    """跑回合并回调 on_step()（每控制步一次，用于渲染）。返回结局计数。"""
    counts = Counter()
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done, status = False, None
        while not done:
            act, _ = model.predict(adapt(obs), deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            on_step()
            done = term or trunc
            status = info['status']
        counts[status] += 1
        print(f'  episode {i + 1}: {status} ({env.ctx.step} steps)', flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', required=True, help='PPO checkpoint .zip')
    parser.add_argument('--stage', default='D', choices=['A0'] + list('ABCD'))
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noisy', action='store_true', help='保留感知噪声与物理随机化')
    parser.add_argument('--video', default=None, help='输出 mp4 路径（离屏渲染）')
    parser.add_argument('--viewer', action='store_true', help='交互式 viewer（需显示环境）')
    parser.add_argument('--cam', default='top_camera',
                        help='相机名（scene_grasp.xml 的 top_camera；free = 自由视角')
    args = parser.parse_args()
    if not args.video and not args.viewer:
        parser.error('需要 --video PATH 或 --viewer 之一')

    import os
    if args.video:
        os.environ.setdefault('MUJOCO_GL', 'egl')   # headless 离屏渲染

    from stable_baselines3 import PPO
    from fr3_grasp_env import Fr3GraspEnv

    model = PPO.load(args.model, device='cpu')
    adapt, mode = make_obs_adapter(args.model)
    print(f'model: {args.model}\nobs 口径: {mode}', flush=True)

    env = Fr3GraspEnv(stage=args.stage)
    if not args.noisy:
        env.stage_params['noise_scale'] = 0.0
        env.stage_params['domain_rand'] = False

    if args.video:
        import cv2
        import mujoco
        h, w, fps = 720, 1280, 20            # 1 控制步 = 0.05 s → 20 fps 实时
        cam_id = env.sim.model.camera(args.cam).id if args.cam != 'free' else -1
        renderer = mujoco.Renderer(env.sim.model, h, w)
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (w, h))

        def on_step():
            renderer.update_scene(env.sim.data, camera=cam_id)
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

        try:
            counts = run_episodes(env, model, adapt, args.episodes, args.seed,
                                  args.noisy, on_step)
        finally:
            writer.release()
            renderer.close()
        print(f'video saved: {args.video}')
    else:
        import mujoco.viewer
        dt = 0.05                              # 与训练的控制周期一致
        with mujoco.viewer.launch_passive(env.sim.model, env.sim.data) as view:
            def on_step():
                view.sync()
                time.sleep(dt)
            counts = run_episodes(env, model, adapt, args.episodes, args.seed,
                                  args.noisy, on_step)
            time.sleep(1.0)                    # 最后一帧停留
    env.sim.close()
    n = args.episodes
    print(f'stage={args.stage} episodes={n}: '
          + ', '.join(f'{k} {v} ({v / n:.0%})' for k, v in counts.items()))


if __name__ == '__main__':
    main()
