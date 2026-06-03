# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import threading as _threading
from collections import Counter
from typing import Any, Iterable, Optional

import ray

from nemo_rl.algorithms.async_utils.interfaces import ReplayBufferProtocol


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
class ReplayBufferImpl(ReplayBufferProtocol):
    """FIFO replay buffer with a staleness window (AReaL-style).

    A single entry corresponds to 1 prompt repeated by
    grpo.num_generations_per_prompt (required to compute per-prompt advantages).

    Each trajectory carries only the weight version that generated it
    (``trajectory_version``). The trainer consumes trajectories first-in,
    first-out from within the staleness window
    ``[current_weight_version - max_age_steps, current_weight_version]``;
    anything older is evicted at sample time.

    There is no per-trajectory "target training step" reservation. The trainer
    takes whatever valid trajectories exist, so it never stalls on a single slow
    straggler while fresher, still-in-window trajectories sit unused. Whatever
    staleness surfaces is corrected by the off-policy loss (importance-sampling
    correction / CISPO), exactly as in AReaL and prime-rl.
    """

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self.trajectories: list[dict[str, Any]] = []
        # weight-version used to generate each trajectory (FIFO/insertion order)
        self.trajectory_versions: list[int] = []
        self._lock = _threading.Lock()

    def add(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
    ) -> str:
        """Add a per-prompt trajectory group.

        Args:
            trajectory: data dict
            weight_version: version of the model weights used for generation
        """
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            return "success"

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        with self._lock:
            return {
                "total_trajectories": len(self.trajectories),
                "trajectory_versions": list(self.trajectory_versions),
                "max_size": self.max_size,
            }

    def _remove_indices(self, indices: Iterable[int]) -> None:
        """Remove trajectories at the given indices."""
        for idx in sorted(indices, reverse=True):
            self.trajectory_versions.pop(idx)
            self.trajectories.pop(idx)

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample ``num_prompt_groups`` per-prompt trajectory groups, FIFO.

        Trajectories older than the staleness window
        ``[current_weight_version - max_age_steps, current_weight_version]`` are
        evicted first. If fewer than ``num_prompt_groups`` valid trajectories
        remain, returns None so the trainer waits for generation to catch up
        (this is genuine generation lag, not a reservation stall).

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None
            if insufficient data.
        """
        with self._lock:
            if not self.trajectories:
                return None

            total_before = len(self.trajectories)

            # Evict trajectories that have aged out of the staleness window.
            min_valid_version = max(0, current_weight_version - max_age_steps)
            stale = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if v < min_valid_version
            ]
            if stale:
                self._remove_indices(stale)
                print(
                    f"🗑️ Evicted {len(stale)} stale trajectories "
                    f"(gen version < {min_valid_version})"
                )

            available = len(self.trajectories)
            if available < num_prompt_groups:
                print(
                    f"Insufficient valid groups: have {available}, "
                    f"need {num_prompt_groups}. Waiting for buffer to fill."
                )
                return None

            # FIFO: consume the oldest still-valid trajectories first. They are
            # the closest to eviction, so consuming them first minimizes wasted
            # generation while staying inside the staleness window.
            selected = list(range(num_prompt_groups))

            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )
            sampled_items = [self.trajectories[i] for i in selected]
            self._remove_indices(selected)

            print(
                f"✅ Sampled {len(selected)} groups FIFO "
                f"(gen-version counts: {Counter(sampled_weights)}, "
                f"avg age: {avg_trajectory_age:.2f} steps); "
                f"buffer {total_before} → {len(self.trajectories)}"
            )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }

    def size(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.trajectories.clear()
            self.trajectory_versions.clear()


@ray.remote  # pragma: no cover
class ReplayBuffer(ReplayBufferImpl):
    pass


# WIP: DO NOT USE - This class is WIP and may be changed without notice, please DO NOT USE it.
# Will be replaced by TQReplayBuffer once TQ is ready.
@ray.remote  # pragma: no cover
class ReplayBufferNew(ReplayBufferImpl):
    """Staleness-window replay buffer with freshest-first sampling.

    -- WIP: DO NOT USE --
    This class is WIP and may be changed without notice, please DO NOT USE it.

    Differences from ReplayBuffer:
    - max_staleness is fixed at construction (instead of being passed per
      sample() call as max_age_steps).
    - sample(): selects trajectories in freshest-first order (default) or FIFO
      order, controlled by the sample_freshest_first flag, from whatever remains
      in the buffer after eviction.
    """

    def __init__(
        self, max_size: int, max_staleness: int, sample_freshest_first: bool = True
    ):
        super().__init__(max_size)
        if max_staleness < 0:
            raise ValueError(f"max_staleness must be non-negative, got {max_staleness}")
        self.max_staleness = max_staleness
        # will move to StalenessSampler when we implement it
        self.sample_freshest_first = sample_freshest_first

    def _evict(self, current_weight_version: int) -> None:
        """Evict rows where trainer_version - weight_version > max_staleness.

        Must be called with self._lock held.
        """
        min_valid = current_weight_version - self.max_staleness
        stale = [i for i, v in enumerate(self.trajectory_versions) if v < min_valid]
        self._remove_indices(stale)

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample num_prompt_groups trajectories, freshest-first.

        Will evict stale rows before sampling, so we will get [current_weight_version - self.max_staleness, current_weight_version] valid trajectories.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None.
        """
        with self._lock:
            self._evict(current_weight_version)

            if not self.trajectories:
                return None

            all_indices = range(len(self.trajectory_versions))
            if self.sample_freshest_first:
                all_indices = sorted(
                    all_indices,
                    key=lambda i: self.trajectory_versions[i],
                    reverse=True,
                )

            if len(all_indices) < num_prompt_groups:
                print(
                    f"Insufficient trajectories: have {len(all_indices)}, "
                    f"need {num_prompt_groups}. Waiting."
                )
                return None

            selected = all_indices[:num_prompt_groups]
            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )

            sampled_items = [self.trajectories[i] for i in selected]
            self._remove_indices(selected)

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }
