# FR3 视觉抓取系统（ROS 2 / MoveIt 2 / MuJoCo / 强化学习）

基于 **ROS 2 Humble + MoveIt 2 + MuJoCo** 的 Franka FR3 机器人抓取系统，包含两条完整主线：

1. **视觉定位与抓取放置**：RGB-D 相机 → OpenCV 颜色检测 → 3D 定位 → MoveIt 2 运动规划 → 物理接触抓取 → 运输放置，端到端全自动闭环（不依赖强化学习即可完整运行）；
2. **强化学习课程训练**：针对接触抓取阶段（下降→对准→闭合→抬升）的 PPO 策略，PBRS 势函数奖励塑形 + 五阶段课程学习 + 行为克隆预热，训练与部署共用同一套观测/动作/IK/成功判据。

当前支持红、绿、蓝三色 3 cm 方块，固定俯拍相机。本项目仅用于仿真。

**技术栈**：Python / C++、ROS 2 Humble、MoveIt 2、MuJoCo 3.x、OpenCV、TF2、Gymnasium、Stable-Baselines3（PPO）、PyTorch、pytest、TensorBoard


<video src="./docs/play_final.mp4" controls width="600"></video>

## 系统架构

```
RGB-D 相机 ──OpenCV 颜色分割──▶ 3D 目标定位（fr3_link0 基座坐标系）
                                    │
                     MoveIt 2 自由空间规划（接近 / 运输 / 放置）
                                    │
         抓取阶段：脚本直线轨迹 或 RL 策略（下降 → 对准 → 闭合 → 抬升）
                                    │
                  ROS ↔ MuJoCo 物理线程桥接（逐步写 ctrl）
```

任务状态机：`DETECTING → OPENING → APPROACHING → 抓取（脚本/RL）→ ATTACH → TRANSPORTING → PLACING`

***

## 第一部分：视觉定位与抓取放置（无 RL 完整管线）

### 1. 视觉感知

- **RGB-D 对齐 + 近似时间同步**：RGB / depth / CameraInfo 三话题 `ApproximateTimeSynchronizer`（25 ms 容差）同步采集，校验光学帧一致性与深度编码；
- **颜色分割与 3D 反投影**：HSV 颜色分割生成掩码 → 相机内参 K 反投影至相机系 3D 点 → TF2 变换到机器人基座系 `fr3_link0`，感知全程不依赖仿真真值；
- **工作空间过滤**：按已知桌面高度与方块尺寸过滤视野外的彩色干扰（如彩色收纳盒），并按面积保留各颜色最大连通域；
- **输出**：各颜色目标位姿 `/perception/<color>/pose`、2D 检测框、RViz Marker、标注图像。

### 2. 视觉定位精度（实测）

自动化验收脚本 `scripts/validate_demo.py` 以 `/simulation/grasp_metrics` 的 MuJoCo 物体真值为基准，计算视觉位姿与真值的欧氏距离（同一基座坐标系），验收阈值 < 5 mm：

| 目标颜色 | 定位误差（欧氏距离）     |
| ---- | -------------- |
| 红    | 0.02 – 1.12 mm |
| 绿    | 2.08 mm        |
| 蓝    | 1.57 mm        |

逐分量实测表明误差为**确定性偏置而非随机噪声**：三色 xy 偏置分别约 (−0.02, 0.00)、(+0.68, +1.99)、(−1.57, −0.18) mm，z 分量误差为 0，合并散布 σ ≈ 0.9–1.0 mm。成因是不同颜色的掩码分割像素差异导致质心偏移与深度量化。

该实测直接闭环用于 RL 训练的感知噪声建模（见第二部分）：σ\_xy = 1.5 mm 覆盖实测最大偏置，σ\_z = 0.5 mm 留深度离散余量。

### 3. 运动规划与物理抓取

- **MoveIt 2** 负责自由空间运动：接近（方块上方 12 cm、末端朝下、yaw 对齐）、抓取后运输、放置与撤回；
- **多维可验证的抓取成功判据**：TCP 相对方块起始高度的抬升 ≥ lift\_height；实际双指总开合宽度 ∈ \[0.5, 1.2] × 方块尺寸；实际开宽 − 指令开宽 ≥ 3 mm（实际夹持验证）；全程无安全中止触发；
- **成功上报后持续校验**：任务报告成功后继续检查 2 s 仿真时间的双指接触保持，杜绝"假成功"（服务返回值仅代表"已开始"，不作为成功依据）。

### 4. ROS ↔ MuJoCo 桥接与可靠性工程

- **物理线程逐步写控制**：轨迹插值从指令线程下沉到物理线程逐步写 ctrl（`active_traj` 机制），消除指令线程与物理步进失同步导致的控制步跳变，根治运输阶段掉块；
- **接触参数调优**：MuJoCo 椭圆锥摩擦模型（`cone="elliptic" impratio="10"`）抑制接触蠕滑，多次实验稳定夹持通过运输；
- **在线持握监控**：基于 `/joint_states` 双指指宽判断掉块（指宽 < 方块尺寸一半即判定脱落），`threading.Event` 线程安全通信，掉块即刻中断并上报 FAILED；
- **接口健壮性**：malformed 轨迹、并发 action、忙时 reset 均按预期拒绝；取消轨迹后保持实际关节位置。

### 5. 端到端验收结果

`scripts/validate_demo.py` 自动化验收（Ubuntu 22.04 + ROS 2 Humble + MuJoCo 3.13.0）：

| 项目                                         | 结果                       |
| ------------------------------------------ | ------------------------ |
| 视觉定位误差（红/绿/蓝）                              | 0.02 – 2.1 mm（阈值 < 5 mm） |
| 红/蓝目标连续抓取                                  | 全部成功                     |
| 物体实际抬升                                     | 86.8 – 90.9 mm           |
| 抬升后双指接触保持                                  | 通过（连续检查 2 s 仿真时间）        |
| malformed 轨迹 / 并发 action / 忙时 reset / 取消轨迹 | 全部按预期处理                  |
| 内参、反投影、轨迹、URDF/MJCF TCP 回归测试               | 全部通过                     |

验收脚本生成 `report.json`、launch 日志与抓取前后图像；关节状态约 68–88 Hz，仿真速度约 0.68–0.88 倍实时（软件渲染）。

***

## 第二部分：RL 课程训练（接触抓取策略）

### 1. 问题定义

RL 只负责接触抓取阶段（下降 → 对准 → 闭合 → 抬升约 10 cm），自由空间运动仍由 MoveIt 完成。策略每步自主决定移动方向，不预设人为轨迹。

- **动作空间（5 维，\[-1, 1]）**：笛卡尔增量 (Δx, Δy, Δz, Δyaw) + 夹爪开合宽度，经**阻尼最小二乘差分 IK** 映射为 7 关节位置伺服指令，零空间项将冗余自由度拉向 home 姿态防止关节漂移；
- **观测设计（部署一致性优先）**：actor 只使用部署时可获取的本体量；感知目标位姿在回合内固定不变（俯拍相机在抓取时被夹爪遮挡，部署时无法获取实时目标位姿）；回合结束条件只用本体量判断，仿真真值仅用于奖励计算与评估；
- **训练不跑视觉**：用"仿真真值 + 实测校准的感知噪声"代替感知输出（渲染比物理仿真慢 1–2 个数量级），噪声幅度按第一部分的感知误差实测设置。

### 2. 奖励设计（PBRS 势函数塑形）

- 稠密引导项**只有**势函数差分 `r = γ·Φ(s′) − Φ(s)`，Φ 由 reach / yaw / close / grasp / lift 五个 ∈ \[0,1] 的分量加权组成（Φ\_max = 4.4）；
- 回合结局只由稀疏项决定：成功 +20、越界/碰桌 −5、掉落 −3，另有推移与动作突变小惩罚；
- PBRS 保证最优策略不变，**悬停和故意失败均赚不到塑形分**，从机制上杜绝 reward hacking；
- PPO 的 gamma 与势函数 γ 强制一致（训练启动时校验，不一致直接退出）。

### 3. 五阶段课程学习（自动晋升）

| 阶段       | 目标采样范围 | yaw            | 感知噪声 | 域随机化 | 干扰方块             |
| -------- | ------ | -------------- | ---- | ---- | ---------------- |
| A0       | 小范围    | 0°             | 0.5× | ✗    | 远处（工作区外）         |
| A        | 小范围    | 0°             | 0.5× | ✗    | 远处               |
| B        | 全范围    | 0°             | 1.0× | ✓    | 远处               |
| C（3 档递进） | 全范围    | ±15°→±30°→±45° | 1.5× | ✓    | 远处               |
| D（2 档递进） | 全范围    | ±45°           | 1.5× | ✓    | 邻近间距 0.10→0.06 m |

- **晋升判据**：滚动 500 回合成功率 ≥ 90%，升级后窗口清零重新积累，全程无需人工干预；
- **域随机化**：方块质量 0.018–0.035 kg、方块摩擦 0.9–1.4、指尖摩擦 0.8–1.2；
- **感知噪声按实测校准**：σ\_xy = 1.5 mm、σ\_z = 0.5 mm、σ\_yaw = 2°（高斯截断）。

### 4. 训练管线：行为克隆预热 + PPO 微调

纯随机探索的对照实验表明其无法覆盖抓取所需的长动作序列（33 万步成功率为 0，timeout 100%），据此确立 BC + PPO 路线：

- **BC 预热**：脚本基线（复现部署端直线下降/闭合/抬升）采集 200 个成功回合，同步更新归一化统计量后训练 actor 初始化策略；critic value MSE 从 46.96 收敛至 0.019，actor MSE ≈ 0.01；
- **PPO 微调**：12 并行环境（`SubprocVecEnv`），MLP \[512, 256, 128]，n\_steps=256，batch 1024，lr 3e-4，ent\_coef 0.005，target\_kl 0.03；
- **策略崩坏三重根因的定位与修复**（本部分核心调试工作）：
  1. PPO 默认 `log_std_init=-0.5` 引入的动作噪声（std≈0.6）直接摧毁 BC 预训练策略 → 通过带噪采样实验实测确定安全阈值 std ≤ 0.22，最终取 `log_std_init=-2.5`（std≈0.08）；
  2. BC 采集时未更新 VecNormalize 统计，保存的归一化参数为空 → PPO 加载后观测分布突变引发策略崩坏；
  3. 低方差观测维度（如 yaw 相关余弦值）被标准归一化放大数千倍 → 微调链路全程关闭观测归一化。

### 5. Sim-to-Real 一致性

- 观测、动作、IK、安全约束、成功判据统一封装在共享模块 `fr3_vision_grasp/grasp_policy.py`，**训练（不依赖 ROS）与 ROS 部署 import 同一份实现**；
- 策略可导出为 numpy 权重（`.npz`），ROS 部署端 `NumpyPolicy` 纯 numpy 推理，不引入 torch 依赖；
- 训练与部署共用同一份 mujoco / numpy 环境，保证 IK 与推理数值一致。

### 6. 分层验证体系与训练成果

| 验证层                         | 内容                              | 结果                                        |
| --------------------------- | ------------------------------- | ----------------------------------------- |
| 奖励方向自检（`rl/smoke_check.py`） | 脚本/随机/上飞/悬停/下降悬停 5 类策略 × 11 项信号 | 全 PASS（漏洞策略总奖励 ≤ 1，无法赚取塑形分）               |
| BC 自检门禁                     | 确定性 + 带噪采样各 5 回合                | 双双 5/5                                    |
| 训练管线冒烟                      | 8192 步                          | ep\_rew\_mean +18.3（脚本基线参考 ≈ +18.7）       |
| pytest 回归                   | 环境 / IK / 奖励 / 课程晋升             | 24 项全通过                                   |
| 课程训练                        | A0→A→B→C→D 全档自动晋升               | 各档均满足滚动 500 回合 ≥ 90% 成功率判据，最终在 D 档收敛后策略稳定 |

最终策略在最高难度 D 档（全范围 + ±45° yaw + 1.5× 感知噪声 + 域随机化 + 干扰方块 6 cm 间距）下稳定达到晋升判据成功率，完成全课程收敛。

***

## 运行

### 视觉定位与抓取放置（无 RL 完整管线）

**环境**：Ubuntu 22.04 + ROS 2 Humble。项目根目录执行：

```bash
bash scripts/setup_ubuntu.sh
bash scripts/run_demo.sh              # 默认红色；color:=green / color:=blue 换色
```

另一终端触发抓取任务：

```bash
source scripts/env.sh
ros2 service call /pick/start std_srvs/srv/Trigger '{}'
ros2 topic echo /pick/status --qos-durability transient_local
```

调用 `/pick/reset` 后再开始下一次抓取。修改代码后执行 `bash scripts/build.sh` 重建。

自动化验收（生成 report.json 与抓取前后图像）：

```bash
python3 scripts/validate_demo.py
```

### RL 训练与评估（不依赖 ROS）

```bash
source .venv-fr3/bin/activate
python rl/smoke_check.py              # 奖励方向自检（11 项全 PASS 才继续）
python rl/train_ppo.py                # 正式训练（默认 12 env，课程自动晋升）
python rl/eval_policy.py              # 评估与导出
```

多核机器训练前 `export OMP_NUM_THREADS=8`。详细流程见 [RL 训练指南](docs/rl_training_guide.md)，设计决策见 [rl\_grasp\_plan.md](docs/rl_grasp_plan.md) 与 [rl\_reward\_fix\_plan.md](docs/rl_reward_fix_plan.md)，接口与验证细节见 [使用指南](docs/usage.md)。

### 可视化运行

**RL 策略回放（纯 MuJoCo，无 ROS）**——`rl/play_policy.py` 加载训练 checkpoint 在场景中演示训练成果：

```bash
source scripts/env.sh
# 交互式 viewer：GUI / 桌面环境，可鼠标拖动视角，实时速率演示
python rl/play_policy.py --model rl/runs/e2_bc_v4/checkpoints/ppo_final.zip \
    --stage D --episodes 5 --viewer
# 视频输出：headless 容器（无显示环境），EGL 离屏渲染 mp4
MUJOCO_GL=egl python rl/play_policy.py --model rl/runs/e2_bc_v4/checkpoints/ppo_final.zip \
    --stage D --episodes 3 --video rl/runs/e2_bc_v4/play_final.mp4
```

`--stage D` 为最难课程档（±45° yaw + 域随机化 + 干扰方块 6 cm 间距）；默认与评估同口径（关闭感知噪声与随机化），加 `--noisy` 开启扰动；换 `--stage A` 可对比简单档。

**全流程可视化（视觉感知 + 运动规划 + 抓取放置）**——一条命令启动 MuJoCo 场景、RViz、相机画面、感知与 pick 节点（等价于上文的 `run_demo.sh`）：

```bash
source scripts/env.sh
ros2 launch fr3_vision_grasp fr3_sim.launch.py color:=red
```

待 MoveIt 就绪后，另一终端触发抓取，随后在 pick 节点日志中观看状态机流转 `DETECTING → OPENING → … → SUCCEEDED`：

```bash
source scripts/env.sh
ros2 service call /pick/start std_srvs/srv/Trigger '{}'
```

调用 `/pick/reset` 后可再次开始（换色改 launch 参数 `color:=green/blue`）。注意：此链路的抓取段当前为脚本轨迹；RL 策略已导出至 `rl/runs/e2_bc_v4/policy.npz`（D 档 100 回合 100% 成功，numpy 前向与训练一致性 max|Δ| < 1e-6），接入 pick.py 部署为下一步工作。

## 目录结构

```
franka_ros2_ws/src/fr3_vision_grasp/   # ROS 2 包：perception / pick / bridge / grasp_policy 等
mujoco_simulations_fr3/                # MuJoCo 模型（fr3.xml / scene_grasp.xml）
rl/                                    # 训练：gym 环境 / PPO / BC 预热 / 课程 / 自检 / 测试
scripts/                               # 安装 / 启动 / 自动化验收（validate_demo.py）
docs/                                  # 设计文档、训练指南与验证记录
```

## License & Sources

本项目基于 [Apache-2.0](LICENSE) 许可。

机器人描述基于 [Franka Robotics/franka\_description](https://github.com/frankarobotics/franka_description)，MuJoCo 模型基于 [MuJoCo Menagerie/franka\_fr3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3)。上游许可、版权声明及附加条款均已保留，详见 [Third-Party Notices](THIRD_PARTY_NOTICES.md)。

本项目与 Franka Robotics 无隶属关系，亦未获其官方认可。
