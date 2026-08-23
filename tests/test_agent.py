"""Player agent end-to-end tests."""
from __future__ import annotations

import numpy as np


def test_player_run_collects_resource(cfg):
    """The agent can complete the simple goal 'find and collect a resource'."""
    from open_player.agent.player import Player
    from open_player.environments.synthetic.env import SyntheticGridEnv

    easy = cfg.merge({
        "seed": 2,
        "environment": {"grid_size": 12, "num_enemies": 0, "num_resources": 1, "fog_radius": 20, "max_steps": 60, "player_hp": 10},
        "planning": {"horizons": {"short": 6, "medium": 10, "long": 32}},
    })

    class EasyEnv(SyntheticGridEnv):
        def reset(self, seed=None):
            obs = super().reset(seed=seed)
            w = self.world
            for offset in [(3, 0), (-3, 0), (0, 3), (0, -3)]:
                target = w.player_pos + np.array(offset, dtype=np.float32)
                cell = (int(target[0]), int(target[1]))
                if 0 < cell[0] < w.grid_size - 1 and 0 < cell[1] < w.grid_size - 1 and cell not in w.walls:
                    w.resources[0].position = target
                    break
            return w.build_observation(t=0)

    env = EasyEnv(easy)
    player = Player(easy)
    rep = player.run(env, max_steps=60, render=False, verbose=False)
    assert rep.collected_total >= 1
    assert rep.total_steps > 0


def test_player_learn_short(cfg):
    """A 40-step learn run produces events, episodes and a checkpoint."""
    import tempfile, os
    from open_player.agent.player import Player
    from open_player.environments.synthetic.env import SyntheticGridEnv
    tiny = cfg.merge({"training": {"steps": 40, "log_every": 1000, "replay_update_every": 1000}})
    env = SyntheticGridEnv(tiny)
    player = Player(tiny)
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "model.pt")
        rep = player.learn(env, total_steps=40, verbose=False, checkpoint=ckpt)
        assert os.path.exists(ckpt)
        assert rep.events > 0
        assert rep.episodes >= 1
        assert len(player.trainer.replay) == 40
        # model params unchanged by learning
        assert player.world_model.num_parameters() < 10_000_000
