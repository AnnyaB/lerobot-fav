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
    """Build only the state needed by the RTC snapshot/action boundary.

    These tests exercise the production RTCInferenceEngine methods directly; no
    alternate inference adapter or shadow implementation is used.
    """
    engine = RTCInferenceEngine.__new__(RTCInferenceEngine)
    InferenceEngine.__init__(engine, task="pick")
    engine._obs_lock = Lock()
    engine._obs_holder = {"obs": None, "robot_type": "test"}
    engine._reset_epoch = 0
    engine._action_queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=3))
    return engine


def _chunk(offset: float = 0.0) -> torch.Tensor:
    return torch.arange(12, dtype=torch.float32).reshape(4, 3) + offset


def test_worker_snapshot_pairs_observation_with_one_queue_cursor():
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

    returned = engine.get_action(None)
    torch.testing.assert_close(returned, processed[0])

    obs1 = {"observation.state": torch.tensor([1.0])}
    engine.notify_observation(obs1)
    second = engine._capture_state_snapshot()
    assert second is not None
    assert second.observation is obs1
    assert second.queue.action_index == 1
    torch.testing.assert_close(second.queue.original_left_over, original[1:])

    # A live action pop cannot retroactively mutate an already captured snapshot.
    assert first.queue.action_index == 0
    torch.testing.assert_close(first.queue.original_left_over, original)


def test_chunk_replacement_after_notification_is_visible_without_republishing_observation():
    """Regression for caching queue state inside notify_observation().

    A previous inference may finish after the latest observation notification and
    replace the action queue. The next worker snapshot must pair that latest
    observation with the replacement queue, not a queue tail cached at notification
    time.
    """
    engine = _make_engine()
    old = _chunk()
    new = _chunk(1000)
    engine._action_queue.merge(old, old, real_delay=0, task="pick")

    observation = {"observation.state": torch.tensor([5.0])}
    engine.notify_observation(observation)
    before = engine._capture_state_snapshot()
    assert before is not None
    torch.testing.assert_close(before.queue.original_left_over, old)

    # This mirrors the RTC worker's merge critical section: outer engine lock,
    # then ActionQueue's lock.
    with engine._obs_lock:
        engine._action_queue.merge(new, new, real_delay=0, action_index_before_inference=0, task="pick")

    after = engine._capture_state_snapshot()
    assert after is not None
    assert after.observation is observation
    torch.testing.assert_close(after.queue.original_left_over, new)


def test_action_pop_cannot_interleave_inside_snapshot_capture(monkeypatch):
    """Force the dangerous interleaving with Events instead of sleeps."""
    engine = _make_engine()
    original = _chunk()
    engine._action_queue.merge(original, original, real_delay=0, task="pick")
    engine.notify_observation({"observation.state": torch.tensor([2.0])})

    entered_snapshot = Event()
    release_snapshot = Event()
    real_snapshot = engine._action_queue.snapshot

    def gated_snapshot():
        entered_snapshot.set()
        assert release_snapshot.wait(timeout=2)
        return real_snapshot()

    monkeypatch.setattr(engine._action_queue, "snapshot", gated_snapshot)

    result = {}

    def capture():
        result["snapshot"] = engine._capture_state_snapshot()

    returned = {}

    def pop():
        returned["action"] = engine.get_action(None)

    capture_thread = Thread(target=capture)
    capture_thread.start()
    assert entered_snapshot.wait(timeout=2)

    # Capture now owns _obs_lock. get_action() must wait for the same outer lock.
    pop_thread = Thread(target=pop)
    pop_thread.start()

    release_snapshot.set()
    capture_thread.join(timeout=2)
    pop_thread.join(timeout=2)
    assert not capture_thread.is_alive()
    assert not pop_thread.is_alive()

    snapshot = result["snapshot"]
    assert snapshot is not None
    assert snapshot.queue.action_index == 0
    torch.testing.assert_close(snapshot.queue.original_left_over, original)
    torch.testing.assert_close(returned["action"], original[0])
    assert engine._action_queue.get_action_index() == 1


def test_snapshot_epoch_is_bound_to_the_same_capture_boundary():
    engine = _make_engine()
    original = _chunk()
    engine._action_queue.merge(original, original, real_delay=0, task="pick")
    engine.notify_observation({"observation.state": torch.tensor([3.0])})

    snapshot = engine._capture_state_snapshot()
    assert snapshot is not None
    assert snapshot.reset_epoch == 0

    # Mirror reset's clear-and-bump critical section without needing a policy.
    with engine._obs_lock:
        engine._action_queue.clear()
        engine._obs_holder["obs"] = None
        engine._reset_epoch += 1

    assert snapshot.reset_epoch != engine._reset_epoch
    assert engine._capture_state_snapshot() is None
