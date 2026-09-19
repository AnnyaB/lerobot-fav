#!/usr/bin/env python
"""End-to-end MetaWorld validation for eval episode-cardinality semantics."""

from __future__ import annotations

import json

import torch

from lerobot.envs.configs import MetaworldEnv
from lerobot.envs.factory import make_env
from lerobot.processor import (
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.scripts.lerobot_eval import eval_policy
from tests.fixtures.dummy_checkpoint_policy import make_dummy_policy

BATCH_SIZE = 3
MAX_STEPS = 8
REQUESTED_EPISODES = range(1, 8)
TASK = "assembly-v3"


def _make_processors():
    env_pre = PolicyProcessorPipeline(steps=[])
    env_post = PolicyProcessorPipeline(steps=[])
    policy_pre = PolicyProcessorPipeline(steps=[])
    policy_post = PolicyProcessorPipeline(
        steps=[],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return env_pre, env_post, policy_pre, policy_post


def _make_zero_policy():
    policy = make_dummy_policy()
    with torch.no_grad():
        policy.net.weight.zero_()
        policy.net.bias.zero_()
    return policy


def _one_run(n_episodes: int) -> dict:
    cfg = MetaworldEnv(task=TASK, obs_type="pixels_agent_pos")
    env_dict = make_env(cfg, n_envs=BATCH_SIZE, use_async_envs=False)
    env = next(iter(next(iter(env_dict.values())).values()))

    # Keep the probe cheap. rollout() reads this public wrapper attribute to
    # cap its loop; no simulator dynamics or termination logic is modified.
    for subenv in env.envs:
        subenv._max_episode_steps = MAX_STEPS

    env_pre, env_post, policy_pre, policy_post = _make_processors()
    policy = _make_zero_policy()

    try:
        result = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_pre,
            env_postprocessor=env_post,
            preprocessor=policy_pre,
            postprocessor=policy_post,
            n_episodes=n_episodes,
            return_episode_data=True,
            start_seed=1000,
        )
    finally:
        env.close()

    episode_index = result["episodes"]["episode_index"]
    returned_ids = torch.unique(episode_index, sorted=True).tolist()

    assert returned_ids == list(range(n_episodes)), (n_episodes, returned_ids)
    assert len(result["per_episode"]) == n_episodes
    assert int(episode_index.min()) == 0
    assert int(episode_index.max()) == n_episodes - 1

    return {
        "requested_episodes": n_episodes,
        "returned_episode_ids": returned_ids,
        "per_episode_count": len(result["per_episode"]),
        "trajectory_rows": int(episode_index.numel()),
    }


def main() -> None:
    runs = [_one_run(n) for n in REQUESTED_EPISODES]
    print(json.dumps({"task": TASK, "batch_size": BATCH_SIZE, "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
