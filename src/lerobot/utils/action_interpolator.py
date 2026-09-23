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

"""Action interpolation for smoother robot control.

Provides configurable Nx control rate by interpolating between consecutive actions.
Useful with RTC and action-chunking policies to reduce jerkiness.
"""

from collections.abc import Sequence

import torch
from torch import Tensor


def _rotation_vector_groups(action_keys: Sequence[str] | None) -> tuple[tuple[int, int, int], ...]:
    """Return index triplets for LeRobot end-effector rotation-vector features.

    LeRobot's kinematics processors name axis-angle orientation components
    ee.wx, ee.wy, and ee.wz. Bimanual or custom robots may prepend a
    namespace such as l. or r. Only that explicit convention is recognized:
    generic wx/wy/wz fields may denote angular velocity, so treating them as
    orientations would be unsafe.
    """
    if action_keys is None:
        return ()

    if len(set(action_keys)) != len(action_keys):
        raise ValueError("action_keys must be unique")

    key_to_index = {key: index for index, key in enumerate(action_keys)}
    groups: list[tuple[int, int, int]] = []
    for key, x_index in key_to_index.items():
        if key != "ee.wx" and not key.endswith(".ee.wx"):
            continue

        prefix = key[:-2]
        y_key = f"{prefix}wy"
        z_key = f"{prefix}wz"
        if y_key in key_to_index and z_key in key_to_index:
            groups.append((x_index, key_to_index[y_key], key_to_index[z_key]))

    return tuple(groups)


def _rotvec_to_quaternion(rotvec: Tensor) -> Tensor:
    """Convert one rotation vector to a scalar-first unit quaternion."""
    work = rotvec.to(torch.float64)
    angle = torch.linalg.vector_norm(work)
    half_angle = angle * 0.5
    eps = torch.finfo(work.dtype).eps
    scale = torch.where(
        angle > eps,
        torch.sin(half_angle) / angle.clamp_min(eps),
        0.5 - angle.square() / 48.0,
    )
    quat = torch.cat((torch.cos(half_angle).reshape(1), work * scale))
    return quat / torch.linalg.vector_norm(quat)


def _quaternion_to_rotvec(quat: Tensor, dtype: torch.dtype) -> Tensor:
    """Convert one scalar-first quaternion to its principal rotation vector."""
    work = quat.to(torch.float64)
    work = work / torch.linalg.vector_norm(work)

    # q and -q encode the same rotation. Keep w >= 0 so the returned
    # rotation vector uses the principal angle in [0, pi].
    if work[0].item() < 0.0:
        work = -work

    vector = work[1:]
    sin_half_angle = torch.linalg.vector_norm(vector)
    angle = 2.0 * torch.atan2(sin_half_angle, work[0])
    eps = torch.finfo(work.dtype).eps
    scale = torch.where(
        sin_half_angle > eps,
        angle / sin_half_angle.clamp_min(eps),
        torch.tensor(2.0, dtype=work.dtype, device=work.device),
    )
    return (vector * scale).to(dtype=dtype)


def _slerp_rotvec(start: Tensor, end: Tensor, t: float) -> Tensor:
    """Interpolate rotation vectors along the shortest geodesic on SO(3)."""
    q0 = _rotvec_to_quaternion(start)
    q1 = _rotvec_to_quaternion(end)

    # Quaternions double-cover SO(3). Flip the second endpoint when needed
    # so interpolation follows the shorter physical rotation.
    dot = torch.dot(q0, q1)
    if dot.item() < 0.0:
        q1 = -q1
        dot = -dot
    dot = dot.clamp(-1.0, 1.0)

    # Near-identical rotations are numerically better handled by normalized
    # linear interpolation; elsewhere use exact spherical interpolation.
    if dot.item() > 0.9995:
        quat = (1.0 - t) * q0 + t * q1
        quat = quat / torch.linalg.vector_norm(quat)
    else:
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        quat = (
            torch.sin((1.0 - t) * theta) / sin_theta * q0
            + torch.sin(t * theta) / sin_theta * q1
        )

    return _quaternion_to_rotvec(quat, dtype=start.dtype)


class ActionInterpolator:
    """Interpolates between consecutive actions for smoother control.

    When enabled with multiplier N, produces N actions per policy action
    by linearly interpolating between the previous and current action.

    Example with multiplier=3:
        prev_action -> [1/3 interpolated, 2/3 interpolated, current_action]

    This effectively multiplies the control rate for smoother motion.

    Usage:
        interpolator = ActionInterpolator(multiplier=2)  # 2x control rate

        # In control loop:
        if interpolator.needs_new_action():
            new_action = queue.get()
            if new_action:
                interpolator.add(new_action.cpu())

        action = interpolator.get()
        if action:
            robot.send_action(action)

        # Recording stays at the base FPS: only the tick that emits the
        # policy's own action contributes a dataset frame.
        if interpolator.emitted_policy_action:
            dataset.add_frame(...)
    """

    def __init__(self, multiplier: int = 1, action_keys: Sequence[str] | None = None):
        """Initialize the interpolator.

        Args:
            multiplier: Control rate multiplier (1 = no interpolation, 2 = 2x, 3 = 3x, etc.).
            action_keys: Optional action names in tensor order. Standard LeRobot
                end-effector rotation-vector triplets (ee.wx/wy/wz, optionally
                namespaced such as r.ee.wx/wy/wz) are interpolated geodesically
                on SO(3); every other coordinate remains linearly interpolated.
        """
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.multiplier = multiplier
        self._rotvec_groups = _rotation_vector_groups(action_keys)
        self._prev: Tensor | None = None
        self._buffer: list[Tensor] = []
        self._idx = 0
        self._emitted_policy_action = False

    @property
    def enabled(self) -> bool:
        """Whether interpolation is active (multiplier > 1)."""
        return self.multiplier > 1

    @property
    def emitted_policy_action(self) -> bool:
        """Whether the action last returned by :meth:`get` was the policy's own output.

        :meth:`add` stores the policy action last in the interpolated buffer, so this
        is ``True`` exactly on the tick that emits ``buffer[-1]`` and ``False`` on the
        intermediate ticks leading up to it.  Strategies gate dataset recording on it:
        frames then land at ``fps`` regardless of ``multiplier``, and each one stores a
        genuine policy action paired with the observation that produced it.

        Read this *after* :meth:`get` (or after ``send_next_action``) — it describes
        the action already handed out, not the one the next call will return.
        :meth:`needs_new_action` is the question to ask *before* dispatching; reading
        that one instead would record ``buffer[0]``, the least-advanced intermediate.
        """
        return self._emitted_policy_action

    def reset(self):
        """Reset interpolation state (call between episodes)."""
        self._prev = None
        self._buffer = []
        self._idx = 0
        self._emitted_policy_action = False

    def needs_new_action(self) -> bool:
        """Check if a new action is needed from the queue."""
        return self._idx >= len(self._buffer)

    def add(self, action: Tensor) -> None:
        """Add a new action and compute interpolated sequence.

        Args:
            action: New action tensor from policy/queue (already on CPU).
        """
        if self.multiplier > 1 and self._prev is not None:
            self._buffer = []
            for i in range(1, self.multiplier):
                t = i / self.multiplier
                interp = self._prev + t * (action - self._prev)
                for group in self._rotvec_groups:
                    if max(group) >= action.numel():
                        raise ValueError(
                            f"rotation-vector action index {max(group)} is out of bounds "
                            f"for action with {action.numel()} elements"
                        )
                    indices = list(group)
                    interp[indices] = _slerp_rotvec(self._prev[indices], action[indices], t)
                self._buffer.append(interp)
            # The end point is the policy's action itself, appended verbatim rather
            # than computed as ``prev + 1.0 * (action - prev)``, which can land an ULP
            # away.  ``emitted_policy_action`` promises the recorded frame carries the
            # policy's own output, so make that exact.
            self._buffer.append(action.clone())
        else:
            # First step: no previous action yet, so run at base FPS without interpolation.
            self._buffer = [action.clone()]
        self._prev = action.clone()
        self._idx = 0

    def get(self) -> Tensor | None:
        """Get the next interpolated action.

        Returns:
            Next action tensor, or None if buffer is exhausted.
        """
        if self._idx >= len(self._buffer):
            self._emitted_policy_action = False
            return None
        action = self._buffer[self._idx]
        self._idx += 1
        self._emitted_policy_action = self._idx == len(self._buffer)
        return action
