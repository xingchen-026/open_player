# Open Player

低算力、非 Transformer 主导的通用游戏学习智能体基础框架。

- Phase 0：跑通最小、可测试、可训练、可扩展的 Learning Agent Core
  （Synthetic Grid World 完整闭环）。
- Phase 1：Learning Validation —— 证明 Open Player 开始真正通过经验学习，
  而不是主要依赖启发式规则（learned vision / multi-step world model /
  learned change prediction / NeuralSkill / intrinsic exploration /
  transfer evaluation）。
- Phase 1.5：Scientific Validation —— 统一评估协议、5+ seeds 学习曲线、
  世界模型 / NeuralSkill / Intrinsic / Multi-step / Representation
  ablations、STRICT_RGB_MODE 泄漏测试、World B/C transfer 与 adaptation。
  完整实验报告见 docs/phase1_5_validation.md（含未通过的假设，如实报告）。

架构决策已冻结，本项目不引入 Transformer / VLM / LLM / MCTS / 大型 RL /
Vector DB，一切按结构化状态 + 小型神经模块 + 显式记忆 + 世界模型 +
层级规划的原则实现。

--------------------------------------------------------------------------------

## Phase 0 做到了什么

1. 完整数据闭环

   - Environment -> Observation -> WorldState -> WorldModel -> Prediction
     -> Loss -> Optimizer -> Model Update 可以运行，loss 可以下降。
   - python examples/phase0_train.py 即可复现。

2. 冻结的核心数据结构

   - EntityState / BeliefState / Relation / SpatialMemory / WorldState /
     Observation / Event / Episode / Goal / Skill / Action 全部实现。
   - Entity 不按具体游戏硬编码：统一 EntityState + semantic_type 字符串，
     张量布局由 EntitySchema 管理（默认 D_entity=42，由字段配置推导，
     代码中不写死）。
   - WorldState 是结构化状态（entities / beliefs / relations /
     spatial_memory / dynamics / temporal / global / uncertainty 八个显式
     张量），不压缩成单一 embedding。

3. World Model（正式路线的 Phase 0 版本）

   - Representation（卷积 + MLP）+ Multi-step Dynamics（残差 MLP）+
     Change/Event Prediction（二分类头）。
   - 完整 API：predict(state, action) / rollout(state, actions, k) /
     loss(prediction, target) / update(...)。
   - 1-step 预测已实现；rollout 接口支持任意 k 步（潜变量自回归）。
   - 参数量约 1.8M，远低于 5M-10M 预算。

4. 训练：Self-Supervised World Learning

   - 在线训练：每步 (Observation_t, Action_t, Observation_t+1) ->
     WorldState_t / WorldState_t+1 -> 预测 -> Loss -> 反向传播。
   - 最小损失集：entity prediction + spatial prediction + change
     prediction + latent consistency（权重全部在 configs/phase0.yaml）。
   - Replay Buffer（容量/采样可配置）、checkpoint 版本化保存/加载、
     CUDA 自动检测（CPU 同样可运行）。

5. Event System

   - Change Detection -> Event；启发式检测器支持 13 种 primitive event
     （appear / disappear / move / approach / collision / damage / death /
     collect / enter / exit / threat_increase / threat_decrease /
     state_change）。
   - Hierarchical Event Graph：primitive -> composite（compose 接口）->
     Episode；边关系支持 temporal / causal / spatial，Event 携带
     parent_event。

6. Episode

   - Event-boundary + Goal 混合分段：goal 完成/失败或环境 episode 结束
     都会产生边界；Episode 包含 start_state / goal / events /
     skill_trace / action_trace / outcome / failure_analysis / end_state。

7. Memory 四层 API

   - WorkingMemory（近期事件/状态/动作/当前 goal）+ EpisodicMemory
     （可运行）。
   - ProceduralMemory（技能成功率统计）/ SemanticMemory（符号事实计数）
     / SpatialMemoryStore（跨步空间记忆累积）提供接口与简单实现。
   - 没有接向量数据库。

8. Goal 系统

   - External Goal + Intrinsic Motivation（novelty / curiosity / progress /
     threat）+ Current Situation -> Candidate Goals -> Utility Scoring ->
     Selected Goal。
   - Phase 0 使用 utility-based scorer；任务/探索/生存/学习/信息/
     技能改进六类 goal 都已接线。

9. Planner

   - Goal -> Subgoal -> Candidate Skills -> World Model Rollout ->
     Outcome Evaluation -> Select Skill。
   - 启发式候选生成 + 短 rollout + utility 打分；没有 MCTS、没有大搜索；
     接口为将来替换成 learned planner 保留。

10. Adaptive Hierarchical Horizon

    - 三档 short=4 / medium=8 / long=32，通过配置文件控制（代码不写死），
      按 goal 类型自适应选择。

11. Skill（Hierarchical Skill / Option）

    - Skill 接口：can_start / act / should_terminate / predict_outcome /
      update。
    - RuleSkill / HeuristicSkill 作为 Phase 0 基础实现（explore /
      approach_resource / collect / avoid_threat），NeuralSkill 接口保留。
    - Skill 不是固定动作序列。

12. Action / Environment

    - ActionSpec + Action + ActionController；DiscreteActionSpace 完整实现，
      Continuous / Hybrid 接口保留。
    - Synthetic Grid World：Player / Enemy / Resource / Walls / Unknown Area
      （迷雾），行为包括移动、接近、碰撞、收集、威胁、死亡、探索。

13. Player Agent

    - player.learn(environment) 与 player.run(environment) 均已实现；
      内部仍可拆开调用 world_model.predict / memory.store /
      planner.plan / skill.act。
    - Agent 能完成"找到资源并收集"这个简单目标
      （python examples/phase0_demo.py）。

14. 工程要求

    - type hints、docstrings、pytest（45 个用例，覆盖 schema / shape /
      environment / world state / world model / event / memory / skill /
      planner / training / agent）。
    - deterministic seed、config-driven 超参数、checkpoint 版本化、
      CPU 可运行、CUDA 自动检测、logging、可复现配置。

--------------------------------------------------------------------------------

## Phase 0 尚未做到什么（已知边界）

- World Model 只保证 1-step 预测质量；multi-step rollout 可用但未做
  长期一致性训练。
- Event Detector 是启发式的，没有训练 Change Encoder / Boundary
  Detector / Event Transformer。
- Goal Scorer 是 utility 式的，没有 learned scorer。
- Planner 是启发式短 rollout，没有 MCTS / 大规模搜索 / learned planner。
- Skill 只有规则实现；NeuralSkill 只是接口。
- Vision 只有 DummyVisionEncoder（结构化观察直通）；没有真视觉编码器。
- Memory 的 Procedural / Semantic 只是简单统计；没有向量数据库。
- 不追求游戏胜率；Phase 0 验证的是 Prediction Error 下降、Event 检测、
  Goal 完成与 Learning Efficiency 的管线是否成立。
- 合成环境的实体数量、迷雾半径等机制是演示性的，不代表最终游戏接口。

--------------------------------------------------------------------------------

## Phase 1 做到了什么（Learning Validation）

1. P1 Learned Vision（约 1.06M 参数，< 3M）

   - RGB (160x90) -> LearnedVisionEncoder（轻量 CNN，无 ViT/Transformer）->
     学习到的空间特征图 [16,32,32] + 每实体 patch 特征 -> WorldState。
   - 位置/速度等结构化量仍来自环境观测（Phase 1 允许），视觉特征由
     世界模型的预测损失端到端训练：CNN 学到的是"对未来预测有用的"
     表示，目标 detach 防止表示坍缩。
   - DummyVisionEncoder 未动，Phase 0 路径继续可用。

2. P2 Multi-step World Model（1/4/8 步）

   - L_total = L_1step + lambda4 * L_4step + lambda8 * L_8step
     （lambda4=0.5 < 1，lambda8=0.25 < lambda4，全部在 configs/phase1.yaml）。
   - Scheduled Rollout：teacher forcing 0.9 -> 0.2 退火 + rollout_ratio
     控制纯 model rollout 比例（Teacher Forcing -> Mixed Rollout ->
     Model Rollout）。
   - 评估输出 1/4/8 步 entity / spatial / latent / change 误差，并给出
     teacher-forcing 上界；实测 4/8 步误差不爆炸（latent 误差随步数
     缓增，长期预测具备可学习性）。

3. P3 Learned Change / Boundary Prediction

   - LearnedChangePredictor（z_t, action, z_t1 -> change logits +
     boundary score，2 层 MLP，无 Event Transformer）。
   - HybridEventDetector：heuristic event 置信度 = blend * learned +
     (1-blend) * heuristic，learned 信号写入 event metadata；
     HeuristicEventDetector 原样保留。

4. P4 NeuralSkill（< 1M 参数）

   - NeuralSkill = MLP policy + learned termination（Option 接口：
     can_start / act / should_terminate / predict_outcome / update /
     memory / metadata）。
   - 训练：RuleSkill 轨迹收集 -> 成功轨迹过滤 -> behavior cloning。
   - 策略输入包含网格级 novelty/wall/threat 图 + 玩家位置 one-hot
     （不做单一 embedding 坍缩），动作带几何有效性 mask。
   - 实测（开放地图）：BC 准确率 90.7%，探索覆盖率 0.98（rule 教师 1.0，
     random 0.37-0.58）—— 行为真正来自经验。

5. P5 Intrinsic Exploration

   - r_intrinsic = alpha*prediction_error + beta*novelty*decay^visits
     + gamma*information_gain - risk_penalty*threat
     - repetition_penalty*repeat_action（全参数进 config）。
   - visit count 驱动的 novelty 衰减、威胁风险惩罚、重复动作惩罚防止
     "撞墙/追敌/原地重复"。
   - 信号真实参与决策：GoalManager 的 exploration/information goal
     优先级受 intrinsic novelty 与 world-model uncertainty 调制；
     ExploreSkill 的目标单元由 intrinsic utility map 选择。

6. P6 Transfer / Learning Efficiency

   - World A（开放地图、资源聚簇、慢敌人）与 World B（窄走廊、资源
     分散、快敌人）结构不同（仅非 seed 差异）。
   - 训练 A、zero-shot 测 B、B 上 1000 步短适应后复测；
     对比 Random / Rule / Phase 0 Agent / Phase 1 Agent 四类 baseline。
   - ExperimentLogger（CSV + JSONL）记录 step / loss / prediction error /
     intrinsic reward / goal success / coverage / skill success /
     transfer score，形成 learning curve 与 transfer curve。

7. 其它

   - SyntheticGridVecEnv：单进程 N 并行世界（评估与技能数据收集使用）。
   - 总神经核心约 2.9M 参数（vision 1.06M + world model 1.85M +
     change predictor ~30k + NeuralSkill ~0.1M），远低于 15M 预算。
   - player.evaluate / player.train_skill / player.evaluate_transfer
     为新增 API，player.learn / player.run 及全部 Phase 0 内部 API 不变。
   - checkpoint 保存/恢复 world model + vision + change predictor +
     NeuralSkill（版本化，向后兼容 Phase 0 checkpoint）。

--------------------------------------------------------------------------------

## 安装

要求：Python 3.11+，PyTorch，NumPy，PyYAML，pytest。

    # 推荐
    pip install -e ".[dev]"

    # 或最小安装
    pip install torch numpy pyyaml
    # 测试
    pip install pytest

Windows / Linux / macOS 均可；无 CUDA 时自动回落到 CPU。

--------------------------------------------------------------------------------

## 快速开始

Phase 0（保持可用）：

    python examples/phase0_train.py --steps 2000 --checkpoint checkpoints/phase0.pt
    python examples/phase0_demo.py --checkpoint checkpoints/phase0.pt --render
    python examples/phase0_checkpoint.py

Phase 1（Learning Validation）：

    python examples/phase1_train.py --steps 2000 --train-skill      # 训练 + 学习曲线
    python examples/phase1_evaluate.py --checkpoint checkpoints/phase1.pt
    python examples/phase1_transfer.py --steps 2000 --adaptation-steps 1000
    python examples/phase1_demo.py --checkpoint checkpoints/phase1.pt --render

Phase 1.5（Scientific Validation，结果写入 results/，报告见
docs/phase1_5_validation.md）：

    python examples/phase1_5_validation.py --experiment baselines
    python examples/phase1_5_validation.py --experiment curve --max-step 10000
    python examples/phase1_5_ablation.py --experiment skill
    python examples/phase1_5_ablation.py --experiment worldmodel
    python examples/phase1_5_ablation.py --experiment multistep
    python examples/phase1_5_ablation.py --experiment intrinsic
    python examples/phase1_5_ablation.py --experiment vision      # 含 STRICT_RGB_MODE
    python examples/phase1_5_transfer.py --experiment transfer
    python examples/phase1_5_plot.py

测试（Phase 0 + Phase 1 + Phase 1.5 共 93 个用例）：

    python -m pytest

最简 API 使用：

    from open_player.core.config import default_config
    from open_player.agent.player import Player
    from open_player.environments.synthetic.env import SyntheticGridEnv

    cfg = default_config()            # 或 load_config("configs/phase0.yaml")
    env = SyntheticGridEnv(cfg)
    player = Player(cfg)
    report = player.learn(env, total_steps=500, checkpoint="checkpoints/phase0.pt")
    demo = player.run(env, max_steps=150, render=True)

--------------------------------------------------------------------------------

## 目录结构

    open_player/
    ├── pyproject.toml
    ├── README.md
    ├── LICENSE
    ├── configs/
    │   └── phase0.yaml
    ├── open_player/
    │   ├── core/            # types / schema / state / specs / config
    │   ├── observation/     # encoder 接口 + DummyVisionEncoder
    │   ├── world/           # representation / dynamics / model / uncertainty
    │   ├── tracking/        # association / tracker（Belief Update）
    │   ├── events/          # types / detector / graph
    │   ├── memory/          # working / episodic / procedural / semantic / spatial
    │   ├── motivation/      # motivation / goals
    │   ├── planning/        # planner / rollout / scoring
    │   ├── skills/          # base / rule / registry
    │   ├── actions/         # specs / controller
    │   ├── environments/    # synthetic: world / env / renderer
    │   ├── training/        # trainer / losses / replay / checkpoint
    │   └── agent/           # player
    ├── tests/                      # 45 (Phase 0) + 32 (Phase 1) 用例
    └── examples/
        ├── phase0_train.py / phase0_demo.py / phase0_checkpoint.py
        ├── phase1_train.py / phase1_evaluate.py
        └── phase1_transfer.py / phase1_demo.py

Phase 1 新增模块：

    open_player/observation/vision.py     # LearnedVisionEncoder (RGB -> WorldState)
    open_player/world/change.py           # LearnedChangePredictor
    open_player/motivation/intrinsic.py   # IntrinsicReward + VisitCounter
    open_player/skills/neural.py          # NeuralSkill + StateFeaturizer
    open_player/training/skill_trainer.py # behavior cloning trainer
    open_player/environments/synthetic/vector_env.py   # SyntheticGridVecEnv
    open_player/environments/transfer.py  # World A / World B
    open_player/evaluation/               # logger / metrics / baselines / benchmark
    configs/phase1.yaml                   # Phase 1 全量配置

--------------------------------------------------------------------------------

## 配置

所有超参数集中在 configs/phase0.yaml（代码中不散落超参数）：

- schema：实体/信念/关系/空间记忆的张量布局（D_entity 由字段推导）
- environment：网格尺寸、敌人/资源数量、迷雾半径、HP、奖励权重
- world_model：各子网络宽度与损失权重
- training：batch size、学习率、replay 容量、更新频率、探索率
- planning：三档 horizon、rollout 步数、候选上限
- goals：动机阈值与 goal 类型权重
- memory / checkpoint：容量与保存策略

CLI 覆盖示例：

    python examples/phase0_train.py --device cuda:0 --seed 7 --steps 5000

--------------------------------------------------------------------------------

## 架构说明

完整架构文档见 docs/architecture.md。要点：

- 依赖方向：environments -> core <- observation/tracking/events/memory/
  motivation/actions/skills <- world <- planning <- training <- agent。
- 模块之间只依赖公开接口（core.specs / 各包 __init__ 导出），不互相
  import 内部实现。
- 所有张量带 batch 维：[B, N, D_entity]、[B, N, N, R]、[B, C, H, W]。
- WorldState 同时保存 entity_ids / semantic_types 列表，支持随时解码回
  结构化对象（规则、事件、技能使用），神经模块直接消费张量。

--------------------------------------------------------------------------------

## 已知限制

- 1-step 世界模型在 1500 步后 entity loss 约 0.3-0.5（归一化特征空间），
  表现为对静态特征的低误差与对快速变化的较高误差；learned change
  预测器对高频语义事件的判别仍在早期。
- 探索策略依赖空间 novelty 通道与威胁回避（Phase 1 已由 intrinsic
  utility 调制，但底层通道仍来自合成环境）；换环境需要重新配参。
- 训练速度约 15-25 steps/s（vision 路径 CPU 约 5 steps/s，GPU 更快；
  瓶颈在 Python 侧环境仿真与状态构建）；按"不追求实时"的目标足够，
  向量化环境已用于评估与数据收集。
- NeuralSkill 目前为 BC 训练（无 RL 微调），且策略输入包含网格级地图
  特征（不依赖真视觉的端到端行为学习仍是后续工作）。
- Episode 分段是 goal 边界 + 环境边界的混合启发式，没有学习式
  segmentation。
- 合成环境不是真实游戏；接入真实环境属于后续 Phase。
- Phase 1 的结论以本仓库的合成世界实验为准，未在商业游戏上验证。

--------------------------------------------------------------------------------

## 后续 Phase（不在本仓库范围）

- 真 Vision Encoder 替换 DummyVisionEncoder
- 训练式 Event/Boundary Detector
- 多步 rollout 一致性训练（世界模型质量）
- NeuralSkill / learned Planner（替换启发式，接口已就位）
- Procedural / Semantic Memory 的深度实现
- 真实环境接入与迁移

--------------------------------------------------------------------------------

## License

MIT（见 LICENSE）。
