"""Tournament agent: two gradient-boosted decision models over engineered features.

    from underdog_gbm.policy import MonopolyAgent
    agent = MonopolyAgent()
    action = agent.choose_action(game, player_id, decision_seed)

Model A (`models/model_a.txt`): trained on raw observation + action encodings; it owns the
auction decision family. Model B (`models/model_b.txt`): trained on engineered valuation /
liquidity / group-dynamics features; it owns every other family. Both models were trained
and evaluated purely on large-scale self-play simulation.

This is the policy, not the competition entry point -- it takes this repository's
internal ``choose_action(game, player_id, decision_seed)`` shape. ``agent.py`` at the
repository root adapts it to the harness contract and is the file the match loads.

``decision_seed`` is accepted for interface compatibility and deliberately unused: the
policy is a deterministic argmax over legal candidates, and it draws from no random
source at all, global or private.
"""

import os

import numpy as np

import engine_shim  # noqa: F401
import feature_lib as fl
import ing_features
from act_lib import _STATIC, SEC_NAMES
from obs_lib import build_obs
from v2_lib import ACT_DIM as ACT_DIM_V2, action_features_v2

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUCTION_SEC = SEC_NAMES.index("auction")


class _Booster:
    def __init__(self, path):
        self.path = path
        self._b = None

    def __call__(self, X):
        if self._b is None:
            import lightgbm as lgb
            self._b = lgb.Booster(model_file=self.path)
        return self._b.predict(X)


class MonopolyAgent:
    """Stateless, seat-agnostic; one instance may play any seat."""

    def __init__(self, name="submission"):
        self.name = name
        self.model_a = _Booster(os.path.join(_HERE, "models", "model_a.txt"))
        self.model_b = _Booster(os.path.join(_HERE, "models", "model_b.txt"))
        self.decisions = 0
        self.forced = 0

    def choose_action(self, game, player_id, decision_seed=0):
        env = fl.unwrap(game)
        pid = int(player_id)
        allowed = sorted(int(a) for a in env.get_allowed_actions(pid))
        if len(allowed) == 1:
            self.forced += 1
            return allowed[0]
        self.decisions += 1
        cand = np.asarray(allowed, dtype=np.int64)
        secs = _STATIC[cand][:, 0]
        if int(secs[0]) == _AUCTION_SEC and (secs == _AUCTION_SEC).all():
            obs = build_obs(env, pid).astype(np.float32)
            X = np.empty((cand.size, obs.size + ACT_DIM_V2), dtype=np.float32)
            X[:, :obs.size] = obs
            X[:, obs.size:] = action_features_v2(env, pid, cand)
            scores = self.model_a(X)
        else:
            scores = self.model_b(ing_features.featurize(env, pid, cand))
        return int(cand[int(np.argmax(scores))])
