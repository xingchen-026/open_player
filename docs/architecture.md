# Open Player 架构说明（Phase 0 + Phase 1）

本文档描述冻结架构在本实现中的落点。设计决策不改动，只说明
"接口在哪里、数据怎么流、将来在哪里替换"。Phase 1 章节只增不改：
Phase 0 的所有结构与 API 保持稳定。

--------------------------------------------------------------------------------

## 1. 总数据流

    SyntheticWorld
        |
        v
    Observation (structured: entities + spatial channels + global features)
        |
        v  ObservationEncoder (DummyVisionEncoder, open_player/observation)
        |
        v  BeliefTracker (Belief Update, open_player/tracking)
        |
        v
    WorldState (open_player/core: 8 个显式张量 + id/type 列表)
        |
        +----> WorldRepresentation (open_player/world) ---> latent z
        |             |
        |             +--> DynamicsModel (z_t, action -> z_t+1)
        |             +--> entity head / spatial head / change head
        |                         |
        |                         v
        |                     Prediction
        |                         |
        |                         +--> Loss (training/losses) -> Optimizer -> Update
        |
        +----> Change Detection -> Event (events/detector, 启发式)
        |             |
        |             +--> EventGraph (temporal / causal / spatial 边)
        |             +--> Episode (goal 边界 + 事件边界)
        |
        +----> Memory (working / episodic / procedural / semantic / spatial)
        |
        +----> Intrinsic Motivation (novelty / curiosity / progress / threat)
        |             |
        |             v
        |        GoalManager: candidates -> utility scoring -> selected Goal
        |
        +----> Planner (planning/)
        |          Goal -> Subgoal -> Candidate Skills
        |             -> WorldModel Rollout -> Outcome Evaluation -> Skill
        |
        v
    Skill (skills/: can_start / act / should_terminate / predict_outcome)
        |
        v
    Action (ActionSpec / ActionController)
        |
        v
    SyntheticWorld  (environments/synthetic)

--------------------------------------------------------------------------------

## 2. 张量布局（configs/phase0.yaml 驱动，代码不写死）

    entities_t      [B, N, D_entity]     N=32, D_entity=42（字段求和）
    beliefs_t       [B, N, D_belief]     D_belief=8
    relations_t     [B, N, N, R]         R=8
    spatial_t       [B, C, H, W]         C=16, H=32, W=32
    dynamics_t      [B, D_dyn]           D_dyn=64
    temporal_t      [B, D_temporal]      16
    global_t        [B, D_global]        32
    uncertainty_t   [B, D_uncertainty]   8

实体字段：position(2) velocity(2) size(1) appearance(8)
semantic_features(16) dynamics_features(12) status(1)，合计 42。

关系字段：distance(1) direction(2) relative_velocity(2) overlap(1)
visibility(1) semantic_relation(1)，合计 8。

WorldState 同时持有 entity_ids / semantic_types 列表；
entity_states() 可按 schema 解码回结构化对象。

--------------------------------------------------------------------------------

## 3. 模块替换点（Phase 0 -> 后续）

    DummyVisionEncoder -> 真 Vision Encoder（同一 ObservationEncoder 接口）
    HeuristicEventDetector -> ChangeEncoder + BoundaryDetector + 事件模型
    UtilityGoalScorer -> learned scorer（GoalManager 不变）
    WorldModelRollout（启发式短 rollout） -> learned planner / MCTS
    RuleSkill -> NeuralSkill（同一 Skill 接口）
    Procedural/Semantic 简单统计 -> 更深实现（接口不变）
    SyntheticGridEnv -> 任意实现 Environment 协议的环境

所有替换都发生在包边界处，core 数据结构和世界模型 API 不动。

--------------------------------------------------------------------------------

## 4. 训练管线（Self-Supervised World Learning）

每个环境 step：

1. 当前 WorldState_t 与 action 输入 WorldModel -> Prediction
2. 观察得到 Observation_t+1 -> BeliefTracker -> WorldState_t+1
3. loss(prediction, target) 计算四类损失并加权：
   entity MSE（按存在概率加权）+ spatial MSE + change BCE + latent MSE
4. Adam 反向传播；grad clip 由配置控制
5. transition 存入 ReplayBuffer；每 replay_update_every 步做一次批量训练
6. UncertaintyEstimator 用实体头误差 EMA 维护不确定性信号
7. Checkpointer 支持版本化保存/加载

--------------------------------------------------------------------------------

## 5. 规划与技能

- Planner.plan(state, goal)：
  1. 按 goal 类型选择 horizon 档（task/survival=4，exploration=8，
     learning/information=32）
  2. SkillRegistry.candidates 过滤 can_start
  3. 静态打分（goal 亲和 + 技能 outcome model + 成功率 + 威胁项）
  4. 对 top-2 候选做 short rollout（默认 4 步），用预测状态评估效用
  5. 选最高分技能，生成 Plan（含 subgoals / expected_utility / scores）
- Skill 每次 act 计数，达到 horizon 或 should_terminate 即重新规划；
  goal 完成后产生 Episode 边界并切换新 goal。

--------------------------------------------------------------------------------

## 6. 确定性

- set_seed(seed) 统一 random / numpy / torch（含 CUDA）
- 环境内部使用 np.random.default_rng(seed)，reset(seed) 可复现
- appearance 特征由 entity_id 的 SHA-256 生成，跨进程稳定
- ReplayBuffer 采样使用独立 torch.Generator

--------------------------------------------------------------------------------

## 7. 性能

- 目标：CPU 可运行、CUDA 自动启用、稳定跑数万步合成环境实验
- Phase 0 实测（本机）：约 25 steps/s（1500 步 < 90 秒）
- Phase 1（vision 路径）：GPU 约 15-20 steps/s，CPU 约 5 steps/s
- 若需提速：减小 batch / 实体数 / horizon / 空间分辨率（配置即可），
  不改架构；SyntheticGridVecEnv 已用于评估与数据收集的批处理

--------------------------------------------------------------------------------

## 8. Phase 1 数据流（Learning Validation）

    RGB_t (160x90)
      -> LearnedVisionEncoder (CNN ~1.06M)
      -> 学习空间特征 [16,32,32] + 实体 patch 特征
      -> WorldState_t（结构化张量不变；位置/速度来自结构化观测）
      -> WorldModel (1/4/8 步预测, scheduled rollout 训练)
      -> Prediction Error -> Intrinsic Reward（含 visit decay / risk /
         repetition 惩罚）-> Goal 选择 + ExploreSkill 目标单元
      -> Planner (heuristic 候选 + learned world model rollout)
      -> Rule / Neural Skill -> Action -> Environment -> RGB_t+1

    LearnedChangePredictor(z_t, a_t, z_t+1)
      -> change logits + boundary score
      -> HybridEventDetector 置信度融合

## 9. Phase 1 模块替换点

    LearnedVisionEncoder -> 更强视觉编码器（同一 ObservationEncoder 接口）
    BC 训练 -> 追加 intrinsic reward fine-tuning（NeuralSkill.update 钩子）
    LearnedChangePredictor -> 多类事件预测（接口不变）
    UtilityGoalScorer -> learned scorer（GoalManager 已接 intrinsic 信号）
    Planner rollout -> 更长/更优搜索（world model 已多步训练）

## 10. Phase 1 关键设计决策

- 视觉目标 detach（vision.detach_targets=true）：CNN 学习"可预测未来"
  的特征，防止自预测表示坍缩；梯度经预测路径回传。
- 多步损失只在序列窗口内以 teacher forcing / mixed / model rollout
  三种模式采样；1 步损失权重最高（1.0 / 0.5 / 0.25）。
- 每个 vision 状态的计算图由其 1 步更新消费后立即 detach，避免二次
  backward；replay 与多步序列存 detach 状态。
- StateFeaturizer 输入不做单一 embedding：实体池化 + global + spatial
  统计 + 网格级地图 + 玩家位置 one-hot，<1M 参数的 MLP 即可学到空间
  探索行为；动作带几何有效性 mask。
- transfer World A / B 通过 wall_density / resource_cluster /
  enemy_move_prob / enemy_attack_prob / grid_size 结构性地不同，
  evaluation 环境与训练环境完全分离，B 的 hidden state 不参与训练。
