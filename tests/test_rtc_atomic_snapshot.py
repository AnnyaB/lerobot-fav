# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic concurrency regressions for temporally atomic RTC snapshots."""

from threading import Event, Lock, Thread

import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference.base import InferenceEngine
from lerobot.rollout.inference.rtc import RTCInferenceEngine


def _make_engine() -> RTCInferenceEngine:
    """Build only production RTC state needed by the snapshot/action boundary."""
    engine = RTCInferenceEngine.__new__(RTCInferenceEngine)
    InferenceEngine.__init__(engine, task="pick")
    engine._obs_lock = Lock()
    engine._obs_holder = {"obs": None, "state_snapshot": None, "robot_type": "test"}
    engine._reset_epoch = 0
    engine._action_queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=3))
    return engine


def _chunk(offset: float = 0.0) -> torch.Tensor:
    return torch.arange(12, dtype=torch.float32).reshape(4, 3) + offset


def test_notification_binds_observation_to_pre_dispatch_queue_cursor():
    engine = _make_engine()
    original = _chunk()
    processed = _chunk(100)
    engine._action_queue.merge(original, processed, real_delay=0, task="pick")

    obs0 = {"observation.state": torch.tensor([0.0])}
    engine.notify_observation(obs0)
    first = engine._capture_state_snapshot()
    assert first is not None
    assert first.observation is obs0
    assert first.queue.action_index == 0
    torch.testing.assert_close(first.queue.original_left_over, original)

    # The rollout contract notifies before get_action(). Consuming that action must not
    # mutate the already-bound snapshot or invalidate its queue generation.
    returned = engine.get_action(None)
    torch.testing.assert_close(returned, processed[0])
    still_first = engine._capture_state_snapshot()
    assert still_first is first
    assert still_first.queue.action_index == 0

    obs1 = {"observation.state": torch.tensor([1.0])}
    engine.notify_observation(obs1)
    second = engine._capture_state_snapshot()
    assert second is not None
    assert second.observation is obs1
    assert second.queue.action_index == 1
    torch.testing.assert_close(second.queue.original_left_over, original[1:])


def test_queue_replacement_fences_old_observation_until_fresh_notification():
    """A new action chunk may not be paired with an observation captured before it existed."""
    engine = _make_engine()
    old = _chunk()
    new = _chunk(1000)
    engine._action_queue.merge(old, old, real_delay=0, task="pick")

    old_obs = {"observation.state": torch.tensor([5.0])}
    engine.notify_observation(old_obs)
    before = engine._capture_state_snapshot()
    assert before is not None
    old_generation = before.queue.generation

    # Mirror RTC's publish critical section. merge() structurally replaces the queue
    # and advances its generation.
    with engine._obs_lock:
        engine._action_queue.merge(new, new, real_delay=0, action_index_before_inference=0, task="pick")

    assert engine._action_queue.get_generation() == old_generation + 1
    assert engine._capture_state_snapshot() is None

    # Only a new control-tick observation can establish the next valid pair.
    new_obs = {"observation.state": torch.tensor([6.0])}
    engine.notify_observation(new_obs)
    after = engine._capture_state_snapshot()
    assert after is not None
    assert after.observation is new_obs
    assert after.queue.generation == old_generation + 1
    torch.testing.assert_close(after.queue.original_left_over, new)


def test_action_pop_cannot_interleave_inside_observation_snapshot(monkeypatch):
    """Force the dangerous notify/get interleaving with Events instead of sleeps."""
    engine = _make_engine()
    original = _chunk()
    engine._action_queue.merge(original, original, real_delay=0, task="pick")

    entered_snapshot = Event()
    release_snapshot = Event()
    real_snapshot = engine._action_queue.snapshot

    def gated_snapshot():
        entered_snapshot.set()
        assert release_snapshot.wait(timeout=2)
        return real_snapshot()

    monkeypatch.setattr(engine._action_queue, "snapshot", gated_snapshot)

    observation = {"observation.state": torch.tensor([2.0])}
    notify_thread = Thread(target=lambda: engine.notify_observation(observation))
    notify_thread.start()
    assert entered_snapshot.wait(timeout=2)

    returned = {}
    pop_thread = Thread(target=lambda: returned.setdefault("action", engine.get_action(None)))
    pop_thread.start()

    # notify_observation owns _obs_lock while snapshot() is gated, so get_action()
    # cannot advance the queue cursor until the observation-bound snapshot is complete.
    release_snapshot.set()
    notify_thread.join(timeout=2)
    pop_thread.join(timeout=2)
    assert not notify_thread.is_alive()
    assert not pop_thread.is_alive()

    snapshot = engine._obs_holder["state_snapshot"]
    assert snapshot is not None
    assert snapshot.queue.action_index == 0
    torch.testing.assert_close(snapshot.queue.original_left_over, original)
    torch.testing.assert_close(returned["action"], original[0])
    assert engine._action_queue.get_action_index() == 1


def test_queue_generation_changes_only_for_structural_mutations():
    engine = _make_engine()
    queue = engine._action_queue
    original = _chunk()

    g0 = queue.get_generation()
    queue.merge(original, original, real_delay=0, task="pick")
    g1 = queue.get_generation()
    assert g1 == g0 + 1

    queue.get()
    assert queue.get_generation() == g1

    queue.merge(original + 10, original + 10, real_delay=0, task="pick")
    g2 = queue.get_generation()
    assert g2 == g1 + 1

    queue.clear()
    assert queue.get_generation() == g2 + 1


def test_reset_epoch_and_queue_generation_both_fence_snapshot():
    engine = _make_engine()
    original = _chunk()
    engine._action_queue.merge(original, original, real_delay=0, task="pick")
    engine.notify_observation({"observation.state": torch.tensor([3.0])})

    snapshot = engine._capture_state_snapshot()
    assert snapshot is not None
    assert snapshot.reset_epoch == 0

    # Mirror reset's single outer critical section without constructing a policy.
    with engine._obs_lock:
        engine._action_queue.clear()
        engine._obs_holder["obs"] = None
        engine._obs_holder["state_snapshot"] = None
        engine._reset_epoch += 1

    assert snapshot.reset_epoch != engine._reset_epoch
    assert snapshot.queue.generation != engine._action_queue.get_generation()
    assert engine._capture_state_snapshot() is None
