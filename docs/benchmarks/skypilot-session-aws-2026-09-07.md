# Reusable SkyPilot session on AWS — 2026-09-07

Status: complete. One explicit `SkyPilotExecutor` session ran four fresh graphs
through one pool-backed native agent: a cold two-task smoke graph, two warm
smoke repeats, and a 20-node serial chain. All 26 AWS work units completed.
One durable allocation record, worker hostname, and Python environment served
the entire sequence.

Warm two-task submission fell from the previous implementation's 76.390-second
median to 4.245 seconds, a descriptive 17.99× speedup and 94.44% reduction.
The 20-node chain fell from 146.564 to 70.224 seconds, a 2.09× speedup and
52.09% reduction. Cold submission was effectively unchanged. This is one
small controlled run, not a general cloud-performance claim.

## Results

`Blocking submit` is monotonic wall time around one fresh graph's
`executor.submit(..., blocking=True)`. Result retrieval is measured separately
and is not included. The blocking pool-apply call and initial `sky check` are
also outside the graph timings. Because an outer session retains its agent,
individual submissions exclude final agent teardown; session close was not
timed separately.

| Graph | AWS blocking submit, s | AWS result fetch, s | One-slot local submit, s | AWS / local |
| --- | ---: | ---: | ---: | ---: |
| Smoke, cold | 179.616 | 0.399 | 3.595 | 49.96× |
| Smoke, warm 1 | 4.359 | 0.430 | 1.130 | 3.86× |
| Smoke, warm 2 | 4.131 | 0.372 | 1.135 | 3.64× |
| Smoke, warm median | 4.245 | — | 1.132 | 3.75× |
| Serial chain, 20 tasks | 70.224 | 4.482 | 27.162 | 2.59× |

The isolated SkyPilot capability check took 10.496 seconds. The blocking
one-worker `pool_apply` call returned in 67.565 seconds, but the immediately
captured pool status was still `PENDING` with no replica. The call itself is
excluded from the table; the cold submission includes the remaining EC2 worker
provisioning/readiness as well as controller, native-agent, and environment
startup. A from-unprovisioned-capacity workflow must account for both phases.

The prior September 6 graph-executor benchmark created a fresh native agent for
each attached graph. Comparing the same small shapes:

| Graph | Prior AWS, s | Session fleet, s | Improvement |
| --- | ---: | ---: | ---: |
| Smoke, cold | 178.512 | 179.616 | none; +0.62% |
| Smoke, warm median | 76.390 | 4.245 | 17.99× faster |
| Serial chain, 20 tasks | 146.564 | 70.224 | 2.09× faster |

The still earlier per-work-unit managed adapter took 216.5 seconds cold and
113.193 seconds warm for the smoke graph. Against that baseline, this run is
17.0% lower cold and 26.7× faster warm. These cross-revision comparisons are
descriptive rather than controlled A/B measurements.

The cold result still includes remaining pool-worker provisioning,
jobs-controller/native-agent acceptance, and the first environment
materialization. The session fleet is designed to amortize those costs, not
remove them from the first graph. The historical one-shot rows also included
per-graph agent cleanup, whereas this run performs it once at session exit.

## Serial dependency latency

Each chain task sleeps for one second and returns worker timing information.
Both AWS and local execution used one logical slot and fresh child processes,
so concurrency does not explain the difference.

| Chain boundary | Session AWS | One-slot local | Prior AWS |
| --- | ---: | ---: | ---: |
| Sum of 20 task bodies | 20.002 s | 20.002 s | 20.002 s |
| First callable after submit | 7.646 s | 0.324 s | 60.433 s |
| First-to-last callable span | 62.069 s | 26.371 s | 71.726 s |
| Sum of 19 dependency gaps | 42.067 s | 6.369 s | 51.725 s |
| Median dependency gap | 2.056 s | 0.333 s | 2.627 s |
| Last callable to submit return | 0.509 s | 0.468 s | 14.406 s |

First-start and final-return figures combine submitter and EC2 wall clocks and
are therefore approximate; the benchmark did not measure their clock offset.
Callable spans and dependency gaps use one worker clock.

Keeping the native agent alive removes most graph-start and graph-end overhead.
The remaining chain cost is now dominated by the command/outcome path between
successors: authenticated S3 mailbox operations, polling, result publication,
input materialization, and fresh-process startup. The benchmark does not
attribute the 2.056-second median gap to any one of those components.

## Method and integrity

| Component | Tested value |
| --- | --- |
| Misen revision | `28b9b6237a9667c80d226f79f52883bfa35a749b` |
| Misen branch | `codex/skypilot-pools` |
| Test project | `priorcomputers/emergent-geometry` v5, revision `440d009fb9082a71371c79b675861d9919c73eed`, with benchmark adaptations |
| SkyPilot | `skypilot-nightly[aws]==1.0.0.dev20260905` |
| AWS worker | One on-demand `c6i.large`, 2 vCPU / 4 GiB declared, `us-east-1` |
| Jobs controller | One isolated `m6i.xlarge` |
| Workspace | Fresh private S3 bucket; 0.2-second coordinator polling |
| Local control | `LocalExecutor`, one CPU slot, 4 GiB declared, fresh disk workspace/environment store |
| Worker Python | AWS 3.13.15; local 3.13.1 |
| Measurement window | 2026-09-07 20:19–20:27 UTC, including setup and cleanup |

Every graph changed its input identity, and the driver rejected any initially
cached work unit. The AWS results contain four distinct run IDs and 26 distinct
logical job IDs. Every task reported the same AWS hostname and environment
path. The serial task PIDs differ, confirming that warm reuse did not share a
user-code interpreter. The protocol archive contains exactly one accepted
native-allocation record for all four graphs.

The session stopped its owned agent and isolated local API lifecycle. The
driver then explicitly removed the borrowed pool and its jobs controller; pool
ownership is intentionally outside normal executor teardown. SDK cleanup
reported no error. Follow-up checks found both exact test-created EC2 instances
terminated, the recorded root volume absent, the private bucket deleted, and no
process from the benchmark API namespace. Final billed cost was not measured.

## Implementation takeaways

1. Session-scoped agents are the right default unit of reuse for related graph
   submissions. They remove fresh native-agent/per-run lifecycle from the warm
   smoke path and preserve environment caches without sharing task interpreter
   state.
2. Provision by bounded resource profile, not by logical work unit. One agent
   executed 26 logical work units across four CPU graphs here; provisioning for
   each node would restore the removed fixed cost.
3. Keep the explicit session visible in the API. It defines agent reuse and
   executor-owned cleanup scope. Borrowed pools and shared controllers can
   remain billable outside it and require their own lifecycle.
4. Optimize steady-state transport next. A lower-latency notification/command
   channel can complement S3 while leaving durable attempts and results in the
   workspace. Dependent one-second tasks still spend about 42 seconds across 19
   handoffs on AWS.
5. Treat cold start separately. Prebuilt images, an already warm controller,
   or explicit agent prewarming may reduce the 179.6-second first graph, but
   this run does not assign a saved-time estimate to those changes.

## Limits and next validation

This benchmark has two warm smoke samples and one chain sample. AWS and local
use different hosts, storage, operating systems, and Python patch releases.
The tasks are sleeps or host inspection, not CPU-throughput workloads. Ratios
are descriptive end-to-end system comparisons.

It validates attached, one-slot, CPU-pool reuse and successful-path cleanup.
It does not validate fan-out/join at width, CPU→GPU→CPU placement, multi-node
Dask, detached coordination, spot interruption, dynamic worker replacement,
injected failures, or failure-path teardown. Those remain separate benchmark
and correctness gates.
