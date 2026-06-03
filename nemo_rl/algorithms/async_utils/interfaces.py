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

from typing import Any, Optional, Protocol


class ReplayBufferProtocol(Protocol):
    """Interface for the replay buffer used in async RL training."""

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
        ...

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups from within the staleness window.

        Trajectories older than ``current_weight_version - max_age_steps`` are
        evicted, then ``num_prompt_groups`` groups are returned FIFO. If too few
        valid trajectories remain, returns None so the trainer waits for
        generation to catch up.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None if insufficient data
        """
        ...

    def size(self) -> int:
        """Return current buffer size."""
        ...

    def clear(self) -> None:
        """Clear the buffer."""
        ...
