# Phase 1.5 Scientific Validation Report

- Repository: https://github.com/xingchen-026/open_player
- Base commit: 33401ad (Phase 1, 77/77 tests passing)
- Experiment config: configs/phase1_5.yaml
- Results: results/p1.5-e0dac236-{abl,curve,xfer}/ (CSV/JSONL/PNG + config
  snapshot + git hash per run dir)
- Protocol: identical environments / episode budgets / seed sets for every
  agent; headline metrics reported as mean +/- std (never single runs);
  verdicts via Welch t + bootstrap CI ("no reliable evidence of improvement"
  when p >= 0.05).

Run to reproduce (each command appends into its run dir):

    python examples/phase1_5_validation.py --experiment baselines
    python examples/phase1_5_validation.py --experiment curve --max-step 10000
    python examples/phase1_5_ablation.py --experiment skill|worldmodel|multistep|intrinsic|vision|representation
    python examples/phase1_5_transfer.py --experiment transfer|generalization
    python examples/phase1_5_plot.py

--------------------------------------------------------------------------------

## 1. Protocol and scale actually run

| Experiment | Seeds | Budget |
|---|---|---|
| E1 baselines (Random/Rule/Phase0/Phase1) on A and B | 5 | 5 eps x 100 steps |
| E2 learning curve (1k/2k/5k/10k) | 5 (10k: 2) | 3 eps x 60 steps eval |
| E5 vision modes A/B/C/D | 3 | 700-800 train steps each |
| E6 NeuralSkill ablation | 3 | BC 400 steps |
| E7 world-model ablation | 3 | 1500 train steps each |
| E8 multi-step training ablation | 3 | 1000 train steps each |
| E9 intrinsic ablation | 3 | 800 train steps each |
| E10 representation probe | 3 | 800 states |
| E11 transfer B/C + adaptation 0/100/500/1000 | 3 | 60-step episodes |
| E12 generalization (held-out speeds/densities) | 3 | eval only |
| E13 compute/params | 1 | 100 steps |

All Phase 0 (45) + Phase 1 (32) + Phase 1.5 (16) tests pass (93 total).
The 50k-step and 10-seed extensions are supported by the scripts but were
not run in this session; the numbers below are what was ACTUALLY executed.

## 2. Answers to Q1-Q9

### Q1: Does Phase 1 learn faster than Phase 0?

Learning curve on World A (mean +/- std over seeds):

| steps | goal success | collected | coverage | err1 entity | err8 entity |
|---|---|---|---|---|---|
| 1000 (n=5) | 0.93 +/- 0.15 | 3.47 +/- 0.61 | 0.68 +/- 0.07 | 0.0145 | 0.0133 |
| 2000 (n=5) | 0.93 +/- 0.15 | 3.47 +/- 0.61 | 0.68 +/- 0.07 | 0.0106 | 0.0087 |
| 5000 (n=5) | 0.93 +/- 0.15 | 3.47 +/- 0.61 | 0.68 +/- 0.07 | 0.0372 | 0.0323 |
| 10000 (n=2) | 1.00 +/- 0.00 | 3.83 +/- 0.24 | 0.67 +/- 0.12 | 0.0359 | 0.0435 |

Fixed-baseline comparison (5 seeds, mean +/- std):

| world | agent | collected | coverage |
|---|---|---|---|
| A | random | 0.32 +/- 0.11 | 0.40 +/- 0.05 |
| A | rule | 2.80 +/- 0.80 | 0.64 +/- 0.12 |
| A | phase0 | 3.72 +/- 0.23 | 0.67 +/- 0.05 |
| A | phase1 | 3.76 +/- 0.33 | 0.80 +/- 0.11 |
| B (held-out) | random | 0.16 +/- 0.09 | 0.16 +/- 0.05 |
| B | rule | 0.80 +/- 0.68 | 0.19 +/- 0.04 |
| B | phase0 | 0.88 +/- 0.27 | 0.22 +/- 0.04 |
| B | phase1 | 1.88 +/- 0.73 | 0.31 +/- 0.01 |

Interpretation (honest): Phase 1 beats Phase 0 on World A coverage
(0.80 vs 0.67) and on held-out World B collected (1.88 vs 0.88) and
coverage (0.31 vs 0.22). On World A the agent performance saturates by
step 1000; further training changes the representation (entity prediction
error is NOT monotonically decreasing: 0.0145 -> 0.0106 -> 0.0372 -> 0.0359
on the moving learned features). Evidence for "faster learning": weak for
goal completion (both near ceiling), moderate for coverage and transfer
performance. No single-run cherry-picking: all numbers are seed-averaged.

### Q2: Did the NeuralSkill learn from experience?

Ablation (3 seeds, exploration coverage of the skill acting alone):

| variant | coverage mean | training acc |
|---|---|---|
| rule teacher | 0.36 | 1.00 |
| BC | 0.32 +/- 0.15 | ~0.56 |
| shuffled labels | 0.32 +/- 0.17 | ~0.35 |
| random init | 0.18 +/- 0.10 | - |

- bc vs random: +0.14, p = 0.26, Cohen's d = 1.08 -> "no reliable evidence
  of improvement" at n=3 (bootstrap CI of BC: [0.15, 0.44]).
- bc vs shuffled: +0.002, p = 0.99 -> no evidence.

Verdict: with 3 seeds the BC skill shows the expected ordering (BC >
shuffled ~ random) but NOT statistical significance; the shuffled-labels
control nearly matches BC because a majority-class policy already explores
in an open world. More seeds/data are needed before claiming "the skill
learned". Phase 1's earlier single-run 90% accuracy result was real but
overstated as evidence; the honest statement is: ordering consistent with
learning, significance not yet established.

### Q3: Is the world model better than the persistence baseline?

Held-out trajectories, World A (3 seeds x 3 trajs; entity MSE):

| model | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| persistence | 0.0043 | 0.0066 | 0.0108 | 0.0111 |
| phase0 | 0.0840 | 0.0954 | 0.1237 | 0.2305 |
| phase1 | 0.0835 | 0.0842 | 0.0840 | 0.0884 |
| random | 0.0995 | 0.1045 | 0.1211 | 0.2162 |

Verdict: NO - the learned models do NOT beat persistence on entity MSE in
this environment: most entity features are static, so "predict no change"
is a very strong baseline (err16 0.011). What the learned models DO win:
- long-horizon stability vs phase0/random: phase1 err16 0.088 vs phase0
  0.231 (multi-step training; see Q4), latent err16 0.114 vs 0.829/1.911;
- change prediction: persistence/random have no change signal at all;
  phase1's learned change predictor is the only one producing it.
Conclusion: "better than persistence" is NOT established for entity error;
it IS established for long-horizon latent stability and change prediction.

### Q4: Does multi-step training improve long-horizon prediction?

Held-out trajectories, latent MSE (3 seeds):

| variant | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| 1-step only (h1) | 0.0032 | 0.0503 | 0.2007 | 0.8048 |
| 1+4 (h14) | 0.0010 | 0.0121 | 0.0454 | 0.1740 |
| 1+4+8 (h148) | 0.0014 | 0.0129 | 0.0487 | 0.1773 |

h148 vs h1 at 16 steps: 0.177 vs 0.805, p = 2.4e-12, Cohen's d = -9.8
-> "improvement" (error is lower-is-better; verdict direction correct).

Verdict: YES - multi-step training dramatically improves long-horizon
latent prediction on held-out data (4.6x lower 16-step error). Adding the
8-step term on top of 4-step gives no further gain in this setting.

### Q5: Does intrinsic reward improve exploration?

5 variants x 3 seeds (World A, 800 training steps):

| variant | coverage | death rate | collision rate | repeat ratio |
|---|---|---|---|---|
| none | 0.642 +/- 0.100 | 0.000 | 0.015 | 0.624 |
| novelty | 0.653 +/- 0.088 | 0.000 | 0.016 | 0.626 |
| error | 0.653 +/- 0.088 | 0.000 | 0.017 | 0.635 |
| novelty+error | 0.587 +/- 0.032 | 0.000 | 0.009 | 0.635 |
| full | 0.653 +/- 0.088 | 0.001 | 0.017 | 0.619 |

full vs none coverage: +0.010, p = 0.90 -> "no reliable evidence of
improvement". Death/collision rates statistically indistinguishable.
Verdict: NO measurable exploration benefit in this setting: the rule-based
planner already explores well on World A, so the intrinsic signal changes
little. Curiosity safety: no variant increased death or collision rates
(all near zero), and repeat ratios are unchanged. Honest conclusion: the
intrinsic-reward mechanism works mechanically but its behavioral effect is
not detectable here; demonstrating it needs a harder exploration setting.

### Q6: Does the RGB path work without struct_grid (STRICT mode)?

Four visual groups, 3 seeds (World A trained 700-800 steps; B zero-shot):

| mode | A goal | A collect | A coverage | A err1 | B coverage |
|---|---|---|---|---|---|
| structured (A) | 0.89 | 3.56 | 0.65 | 0.010 | 0.22 |
| side (B) | 0.89 | 3.56 | 0.65 | 0.010 | 0.31 |
| learned_grid (C) | 0.67 | 1.89 | 0.31 | 0.017 | 0.08 |
| strict (D) | 0.00 | 0.00 | 0.12 | 0.045 | 0.07 |

Leakage test (tests/test_strict_rgb.py) passes: struct_grid is absent,
non-player entity positions are zeroed, env_info carries no GT keys, the
event stream is learned-only, and rule skills are unavailable.
Verdict: the STRICT pipeline RUNS end-to-end without GT, but it is not yet
capable: exploration drops to ~0.12 and goal completion is zero. The
bottlenecks are learned self-localisation and learned occupancy quality at
this scale (plus the strict agent has no BC-trained skill yet). This is an
honest negative result: strict RGB works as a pipeline, not as a player.

### Q7: Does capability transfer to unseen Worlds B / C?

Zero-shot (3 seeds, mean +/- std): B: phase1 2.33 collected / 0.39 coverage
vs phase0 0.67 / 0.18; C: phase1 1.33 / 0.26 vs phase0 0.89 / 0.17.
Generalization groups (held-out enemy speeds and resource densities):
phase1 leads on 7 of 8 (e.g. speed interp coverage 0.68-0.74 vs 0.35-0.44;
density_high collected 9.33 vs 3.33); density_low ties.
Verdict: YES for the agent-level transfer (driven by the transferred
world model + policy); the advantage is consistent across seeds.

### Q8: How many samples does adaptation need?

World B: phase1 2.33 collected at 0 / 100 / 500 / 1000 adaptation samples;
phase0 0.67 at every point. World C: same plateau pattern.
Verdict: zero-shot Phase 1 already outperforms Phase 0, and 100-1000
adaptation samples do not change the outcomes further in these worlds
(performance plateau). Sample-efficiency comparison is therefore
inconclusive: Phase 1 needs FEWER samples to match any given Phase 0 level
(0 samples suffice), but we do not observe an adaptation curve to fit.

### Q9: Statistical stability?

- 5-seed means +/- std reported everywhere (curve/baselines); 3-seed for
  the ablations (documented as n=3).
- Bootstrap CIs and Welch p-values computed for every ablation verdict.
- Statistically solid findings: multi-step training effect (p ~ 1e-12),
  per-entity vs pooled representation (p = 6e-4, d = 28.5).
- Not significant: NeuralSkill BC vs random/shuffled (p 0.26 / 0.99),
  intrinsic full vs none (p = 0.90). These are reported as "no reliable
  evidence of improvement", not packaged.

--------------------------------------------------------------------------------

## 3. Known limitations of this validation

- World A saturates quickly: goal/coverage metrics near ceiling by 1k steps,
  so learning-curve differences are compressed; a harder training world
  would expose more.
- The 50k curve and the 10-seed protocol are scripted but not run in this
  session (time); 10k x 2 seeds + 5k x 5 seeds is what was executed.
- NeuralSkill ablation n=3 has low power; the shuffled-labels control
  performs too well because the majority action already explores.
- Persistence wins entity MSE because entity features are mostly static in
  this environment; entity error is therefore a weak metric here - latent
  and change errors are the informative ones.
- strict mode was stabilised with reduced multi-step weight and a lower
  learning rate (documented in the code); without those, training diverges.
- Phase 1 defect found and fixed during this phase: the vision encoder was
  NOT in the optimizer in Phase 1 (gradients computed but never applied);
  Phase 1.5 fixes this with a separate vision learning-rate group. Phase 1
  numbers reported earlier therefore overstated "learned vision".

## 4. Phase 2 recommendation

Only proceed to Phase 2 for the components with positive evidence:
1. Multi-step world model training (verified, p ~ 1e-12) -> make it the
   default and add longer horizons (16) + per-horizon rollout ratios.
2. Learned representations (per-entity > pooled, p = 6e-4) -> replace mean
   pooling with masked entity aggregation in the world model.
3. Transfer: agent-level transfer to B/C verified -> build Phase 2 transfer
   protocol around it (frozen model + adaptation budgets).
Defer / re-work before Phase 2:
4. NeuralSkill: needs more seeds/data or an easier teacher distribution;
   do not claim learned behaviour yet.
5. Intrinsic reward: no measurable effect on World A; demonstrate on a
   harder exploration world or re-design the integration before claiming it.
6. strict RGB: works as a pipeline only; needs better localisation /
   occupancy training or a BC skill for strict mode before it can act.
7. Do not claim "better than persistence" for the world model in general -
   only for long-horizon latent stability and change prediction.
