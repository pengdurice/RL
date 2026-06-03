# Async GRPO: `target_weight_version` reservation causes avoidable trainer stalls

## Summary

In the async GRPO path, the replay buffer reserves each trajectory for a fixed
*target training step* (`target_weight_version`) at generation time, and the
trainer at step `S` may only consume trajectories tagged `target == S`. When the
batch assigned to step `S` has a slow straggler, the trainer **stalls waiting
for that straggler even though enough fresher, still-in-window trajectories are
already available**. This is a producer/consumer rigidity, not an off-policy
correctness requirement.

We propose dropping the `target_weight_version` tag and moving to AReaL/prime-rl
style **FIFO sampling over a staleness window**, with off-policy correction
(already required for async) handling whatever staleness surfaces.

## Where

- `nemo_rl/algorithms/async_utils/replay_buffer.py` — `ReplayBuffer.sample()`,
  the `intended_indices` / `STALLING` branch.
- `nemo_rl/algorithms/async_utils/trajectory_collector.py` —
  `_calculate_target_weights`, `_get_next_target_for_generation`,
  `_should_pause_for_generation_limits`, `_generating_targets`.

## Root cause

1. The collector reserves one target step per batch (`_get_next_target_for_generation`)
   and stamps every prompt in that batch with the same `target_weight_version`.
   With `max_trajectory_age_steps=1`, trajectories generated at weights `V` are
   always tagged for training step `V+1`.
2. `ReplayBuffer.sample()` first checks the staleness window
   (`valid_indices`), then **further filters** to `target_weight_version == current`
   (`intended_indices`). It stalls (returns `None`) when
   `len(intended_indices) < num_prompt_groups`, *even if* `len(valid_indices) >= num_prompt_groups`.
3. The assignment is a fixed promise made at generation start and is never
   reassigned, so a slow straggler in step `S`'s batch cannot be covered by a
   trajectory that was assigned to step `S+1`.

The tag also drives a **generation throttle**
(`_should_pause_for_generation_limits`): once the next target is reserved, the
collector pauses instead of running ahead to build a backlog.

### Trigger condition

The stall is triggered by **slow / tail-heavy inference**, not by slow training.
If training is the bottleneck, the buffer is always full and the tag never
stalls. The stall is the tail-latency straggler of the batch assigned to the
current step, made un-substitutable by the fixed tag.

## Evidence

From a 10-step async CISPO `lag1` run
(`logs/cispo_mm1_async_lag1_highoffpolicy/slurm-13087.out`):

- The trainer-idle timer `exposed_generation` averaged **25.10 s / step** vs
  `total_step_time` **116.66 s / step** ⇒ **~21.5 %** of wall-clock spent
  waiting (Σ idle 251 s / Σ wall-clock 1167 s). Verified independently:
  **491 failed sample polls × 0.5 s sleep = 245.5 s**.
- Those 491 waits split exactly into two causes:

  | Wait cause | Count | ≈ Time | % wall-clock | Removed by FIFO? |
  |---|---|---|---|---|
  | `STALLING` — buffer had ≥ N **valid** in-window, tag blocked it | **77** | ~38.5 s | **~3.3 %** | **Yes, fully** |
  | `Insufficient valid groups` — buffer genuinely had < N valid | **414** | ~207 s | **~17.7 %** | No (generation-bound) |

- At **every** `STALLING` event, `valid_indices == num_prompt_groups` was already
  satisfied. Example (step 5):

  ```
  version_counts=Counter({4: 29, 5: 3})
  valid_indices: 32/32 trajectories within age window
  🎯 Found 29 trajectories intended for current step 5
  ⏸️ STALLING: Need 32 trajectories for step 5, but only 29 are ready
  ```

  32 valid trajectories existed (29 tagged step 5 + 3 fresher tagged step 6), but
  the trainer stalled for 3 stragglers. FIFO would have proceeded immediately.

- The collector hit the throttle (`...already generated or in progress, pausing`)
  **11 times**, confirming it stops running ahead once targets are reserved.

## Proposal

Adopt AReaL/prime-rl style sampling:

1. Stamp each trajectory only with its generation `weight_version` (drop
   `target_weight_version`).
2. `sample()`: evict trajectories older than
   `current_weight_version - max_age_steps`, then return `num_prompt_groups`
   groups **FIFO** (oldest in-window first). Return `None` only when fewer than
   `num_prompt_groups` valid trajectories remain.
3. Remove the target reservation and generation throttle from the collector;
   bound the producer by the in-flight semaphore + replay buffer `max_size`
   backpressure.
4. Rely on the already-mandatory importance-sampling / CISPO correction to
   handle staleness.

## Expected impact

- **Guaranteed:** removes the ~3.3 % straggler-stall (the 77 `STALLING` polls).
- **Likely larger:** removing the throttle lets the producer run ahead and build
  a backlog, which should also shrink part of the 17.7 % generation-lag waits.
  Magnitude is generation-throughput- and `max_age`-dependent (limited under
  `lag1`); needs measurement.
- This run is fundamentally **generation-bound**; the tag fix removes the tail
  tax but the bulk of idle requires faster/wider generation (more rollout
  capacity, larger `max_age`, partial-rollout / abort-resume).

## Side effects / risks

- **Staleness becomes data-dependent** rather than a fixed per-step promise.
  Mitigated by importance-sampling correction (already required for async).
- **Possible wasted generation:** trajectories that age out of the window are
  evicted unused (trade "wasted trainer wait" for "occasionally wasted rollout").
  FIFO-oldest-first minimizes this.
- **Must keep a capacity/staleness bound** to replace the throttle, else the
  producer over-produces. Handled by `max_size` backpressure + eviction.
- **Lost observability:** per-step "intended for step S" logging goes away;
  `avg_trajectory_age` is still derived from `trajectory_version`.

## Validation plan

- Unit: buffer eviction + FIFO ordering, "insufficient valid groups" wait,
  capacity backpressure.
- E2E: rerun the same `lag1` CISPO config and diff `exposed_generation` /
  `STALLING` count / steps-per-sec against the tagged baseline.
