# SkyPilot warm-cache improvements: v5 benchmark, 2026-09-11

This run measures cached launch commands, concurrent streaming result downloads with per-result local locking, and full declared CPU allocation during environment preparation. SkyPilot worker/session bookkeeping was consolidated. Each WorkUnit still runs in a fresh subprocess. The five changed production files add 73 net lines relative to the previous implementation; `executors/skypilot.py` grows by 14 lines, with all SkyPilot-specific logic remaining there.

| Workflow | Local | Previous persistent AWS | Updated AWS, cold |
|---|---:|---:|---:|
| Serial chain (20 WorkUnits) | 29.3s | 200.9s | 183.3s |
| Fan-out/join (102 WorkUnits) | 36.4s | 270.5s | 239.3s |
| Reduced CUDA sweep (68 WorkUnits) | 33.1s | 343.0s | 314.7s |

The fresh fan-out graph on the same warm pool took **113.5s**, compared with **125.7s** previously. It had **0 new VM launches, 0 environment preparations, and 0 initial result-cache hits**. All WorkUnits completed successfully.

Cold timing includes submission, snapshot staging, the local API/controller, provisioning, environment preparation, execution, result fetch, and executor shutdown. Fan-out's cold comparison adds its first-graph time to teardown measured after the warm repeat; the warm timing excludes that shared teardown. AWS physical termination confirmation is outside these timers.

| Phase (seconds) | Chain | Fan-out | CUDA |
|---|---:|---:|---:|
| Local API startup | 4.271 | 4.276 | 4.269 |
| API-ready → controller-ready | 6.405 | 6.178 | 7.392 |
| VM provisioning / Sky setup / SSH connection | 50.5 | 50.4–53.1 | 50.9–61.0 |
| Environment preparation per VM | 45.8 | 45.1–47.7 | 46.0–64.3 |
| Dispatch → bootstrap, median | 0.080 | 0.075 | 0.075 |
| Warm bootstrap → execute entry, median | 0.251 | 0.256 | 0.266 |
| Imports / payload load, median | 0.022 | 0.022 | 1.331 |
| Execution end → completion observed, median | 0.305 | 0.283 | 0.721 |
| Result fetch | 4.187 | 23.783 | 1.259 |
| Final session teardown | 10.813 | 13.014 | 11.413 |

Warm fan-out bootstrap-to-execute median was **0.259s** (previously **0.451s**). Its result fetch took **20.1s** (previously **24.1s**). New payload transport, interpreter startup, imports, and workspace metadata requests remain on the per-WorkUnit path. Result objects transfer concurrently, while the benchmark still retrieves its output Tasks sequentially.

CPU-worker environment preparation measured 45–48s, versus roughly 57–67s previously. Dependency retrieval and materialization are combined in this phase, so the benchmark does not isolate CPU allocation from network variation. GPU lookahead is unchanged: the T4 launch began 101s after the first CPU launch and took another 125s to connect and prepare. Cold provisioning remains the dominant gap relative to local execution.

## Method and limits

- Workflow repository: `PriorComputers/emergent-geometry`, branch `v5`, pinned to `440d009fb9082a71371c79b675861d9919c73eed`. Same bounded workflows, seeds, and resource declarations as the [previous benchmark](skypilot-persistent-v5-2026-09-11.md).
- AWS `us-east-1`: at most two `m6i.xlarge` CPU VMs (each declared 2 CPUs/8 GiB) and one `g4dn.2xlarge` T4 VM (declared 4 CPUs/16 GiB and one GPU). CPU synthetic functions sleep for one second; their concurrency is not a CPU-throughput measurement.
- CUDA: five variants, eight training steps, batch 8, sequence length 16, model width 32, one layer, four heads, FFN width 64, checkpoints at steps 4/8. Task minimum is 8 GiB per GPU. All five cloud outputs verified CUDA execution, step-8 checkpoints, and finite metrics.
- Local GPU: RTX 3060. Local environment preparation starts with a warm package-download cache; cloud workers are fresh VMs. Different hardware and cold dependency downloads make this a latency comparison, not a normalized throughput comparison.
- Local training was repeated after the test suite finished; that standalone repeat is used in the table. Local chain/fan-out also ran without the test suite competing.
- One cold cloud run per workflow and one warm fan-out repeat. These are observations, not confidence intervals; provisioning/network variation can dominate small differences. Subsecond cross-host phase estimates use wall clocks.
- Timing probes exist only in the private staged benchmark source. The final source adds a Windows-only lock fallback after the timed artifact; the Linux locking/download behavior is unchanged. Initial local runs record the wheel hash current at their start; the final streaming-download artifact was uploaded before any cloud run.

## Validation and retained evidence

The final full test suite passed **1,166 tests**, with 53 skips and eight existing warnings. All 69 affected cloud tests passed; 66 SkyPilot tests passed against nightly 20260911. Source lint, formatting, and type checks passed. Launcher tests cover per-job env-file changes, CPU/GPU settings, fresh processes, and cache misses for changed snapshot/interpreter/bootstrap inputs. Existing snapshot tests cover code-only overlay invalidation and dependency-key changes.

A subsequent pre-commit CI check caught the old `cloudpickle 3.1.0` lock failing on Python 3.14. The lockfile now selects 3.1.2, which includes the [upstream compatibility fixes](https://github.com/cloudpipe/cloudpickle/blob/master/CHANGES.md); the package requirement remains `cloudpickle>=3.0`. Both minimum and latest SkyPilot nightly matrix entries pass 67 tests on Python 3.14, including the cached-launch test now included in CI. This lockfile update does not change the benchmark implementation or its recorded timings.

All **6 benchmark VMs are terminated**, and the temporary S3 bucket is deleted. Archived **3,312 objects (11,576,065 bytes)**; baseline EC2 states are unchanged.

Private evidence is retained at `/home/apoorvkh/workspace/emergent-geometry-bench-20260911/.cache/benchmark-20260911-warm`, including run records, staged source/wheel, phase logs, EC2 timelines, and archived workspace objects. Raw transport logs may contain private signed URLs and should not be published. Shareable aggregates: [skypilot-warm-cache-v5-2026-09-11.json](skypilot-warm-cache-v5-2026-09-11.json).
