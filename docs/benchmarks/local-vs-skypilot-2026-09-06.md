# Local versus SkyPilot/AWS graph execution — 2026-09-06

Status: complete. Eleven planned local measurements plus one supplemental
control executed 574 logical work units; all finished successfully, all began
with zero cached results, and all used the same Misen snapshot as the
[AWS benchmark](skypilot-aws-2026-09-06.md). An independent evidence audit
found no failed tasks, missing outputs, resource assignment errors, or
remaining benchmark processes.

Local execution was substantially faster for every graph. The most useful
peak-concurrency-controlled comparisons are the naturally serial chain and the
two-slot fan-out control: AWS took 5.40 times as long as local for the chain
and 4.42 times as long for the warm fan-out. With the benchmark's configured
12-CPU local budget, AWS took 19.39 times as long as local on the warm fan-out.

These are end-to-end blocking-submit-path comparisons, not isolated
measurements of SkyPilot. Local execution used a disk workspace on an already
running host; AWS used S3 plus SkyPilot's API, controller, native-agent, and
worker lifecycles. CPU and GPU hardware also differed. The ratios therefore
describe the user-visible systems tested, not an intrinsic cloud-versus-local
speedup.

## Headline comparison

`Blocking submit` is monotonic wall time around
`executor.submit(..., blocking=True)`. It includes snapshot/environment
preparation, task execution, and backend-specific work before that call
returns. AWS controller and pool creation are outside the timed boundary; the
cold submission still includes remaining worker readiness and environment
work. These are not from-zero infrastructure timings. Result retrieval is
timed separately. AWS figures reproduce the completed AWS report.

| Workload | AWS blocking submit, s | Local blocking submit, s | AWS / local | Comparison |
| --- | ---: | ---: | ---: | --- |
| Smoke, cold | 178.512 | 49.646 | 3.60× | Descriptive; two local slots, one AWS agent |
| Smoke, warm | 76.390 median | 1.159 | 65.92× | One-slot local control; one supplemental sample |
| Serial chain, 20 tasks | 146.564 | 27.155 | 5.40× | Both naturally peak at 1 |
| Fan-out/join, warm | 303.683 mean | 68.698 | 4.42× | One local sample; both peak at 2 |
| Fan-out/join, warm | 303.683 mean | 15.659 mean | 19.39× | Configured primary local budget, peak 12 versus 2 |
| Training graph, cold | 409.975 | 33.143 | 12.37× | Different hardware, topology, and storage |
| Training graph, warm | 286.050 | 30.685 | 9.32× | Different hardware, topology, and storage |

Including post-submit result retrieval, the AWS/local ratios are 5.58× for
the chain, 4.80× for the peak-concurrency-controlled warm fan-out, 12.62× for
cold training, and 9.61× for warm training. The two-slot local fan-out is one
control sample compared with the mean of two warm AWS samples.

The primary local warm smoke median was 0.626 seconds with its two tasks
assigned to separate CPU slots; AWS took 122.09 times as long. The one-slot
row above is the closer concurrency control. That run was added after the
planned suite had completed; its own durable record passed the same audit,
but it is a single supplemental sample and is not covered by the suite's
completion marker.

## Per-task latency comparison

The synthetic chain and fan-out task bodies each sleep for one second. This
makes scheduling, process startup, dependency transfer, and coordination gaps
visible without claiming to benchmark CPU throughput.

For the fan-out comparison, a local slot is an assigned CPU index and an AWS
slot is a one-task-at-a-time worker agent. Matching two such slots controls
effective callable concurrency, not physical host capacity or performance.

| Metric | SkyPilot/AWS | Local | AWS / local |
| --- | ---: | ---: | ---: |
| Chain: callable span | 71.726 s | 26.503 s | 2.71× |
| Chain: median dependency handoff | 2.627 s | 0.326 s | 8.06× |
| Chain: first callable after submit | 60.433 s | 0.332 s | 182× |
| Chain: last callable to submit return | 14.406 s | 0.320 s | 45.0× |
| Two-slot fan-out: callable span | 216.590 s mean | 68.135 s | 3.18× |
| Two-slot fan-out: median same-slot gap | 2.664 s mean | 0.315 s | 8.46× |
| Two-slot fan-out: p95 same-slot gap | 4.536 s mean | 0.340 s | 13.33× |
| Two-slot fan-out: final-parent-to-join gap | 17.949 s mean | 0.321 s | 55.88× |

The serial chain contains 20.003 seconds of local callable time and 20.002
seconds on AWS. Its 19 local handoffs consume 6.500 seconds in total, compared
with 51.725 seconds on AWS. Because both runs have peak concurrency one, that
gap is not an artifact of greater local parallelism.

The two-slot local fan-out executed exactly 50 branches on each slot and had a
peak of two callables, matching the two AWS agents. Its 102.012 seconds of
callable bodies spanned 68.135 seconds; the warm AWS runs' 102.012-second mean
spanned 216.590 seconds. The local scheduler still starts a fresh subprocess
for every logical work unit, so this comparison does not remove process-start
cost from the local side.

With the local machine's primary 12-CPU budget, the same warm graph had a
14.908-second mean callable span and a 15.659-second mean blocking submit.
This is the configured primary local result, while the two-slot run is the
more useful diagnostic control.

## Full local results

| Run | Work units | Blocking submit, s | Result fetch, s |
| --- | ---: | ---: | ---: |
| Smoke, cold | 2 | 49.646 | 0.176 |
| Smoke, warm 1 | 2 | 0.626 | 0.001 |
| Smoke, warm 2 | 2 | 0.625 | 0.003 |
| Smoke, warm 3 | 2 | 0.627 | 0.004 |
| Smoke, one-slot supplemental | 2 | 1.159 | 0.193 |
| Serial chain | 20 | 27.155 | 0.010 |
| Fan-out, first | 102 | 15.671 | 0.027 |
| Fan-out, warm 1 | 102 | 15.658 | 0.029 |
| Fan-out, warm 2 | 102 | 15.660 | 0.028 |
| Fan-out, two-slot control | 102 | 68.698 | 0.029 |
| Training, cold | 68 | 33.143 | 0.082 |
| Training, warm | 68 | 30.685 | 0.020 |

The cold-to-warm local smoke fell from 49.646 seconds to a 0.626-second
median, a descriptive 98.74% reduction. This is consistent with one-time
snapshot and environment preparation being important, but phase-level timing
was not recorded. “Cold” here means a fresh Misen environment store and
workspace on a previously used host; shared operating-system and
package-download caches could already be warm.

## Heterogeneous training graph

Both local training runs completed the same 68-work-unit, 108-edge topology
and configuration as AWS, using fresh, disjoint seeds: 63 CPU work units and
five GPU work units. All ten selected local results reported `cuda:0`, an
NVIDIA GeForce RTX 3060, CUDA 12.6, final step 8, four metrics rows, and
checkpoints at steps 4 and 8. The local executor admitted at most 12 logical
CPUs, 32 GiB of declared memory, and one GPU concurrently.

| Run | AWS submit, s | Local submit, s | AWS / local | Sum of five training loops: AWS / local, s |
| --- | ---: | ---: | ---: | ---: |
| Cold | 409.975 | 33.143 | 12.37× | 2.496 / 1.727 |
| Warm | 286.050 | 30.685 | 9.32× | 2.489 / 1.753 |

The loop-only difference is about 1.4 times, far smaller than the 9–12-times
end-to-end difference. The measured GPU loops explain little of the end-to-end
difference; work and overhead outside them dominate, but this benchmark cannot
partition CPU preprocessing and postprocessing, imports, CUDA/model
initialization, subprocess bootstrap, scheduling, storage I/O, and lifecycle
costs. AWS used a Tesla T4 plus two CPU agents and one GPU agent on separate
machines; local used an RTX 3060 and one fungible resource budget that could
dynamically pack CPU work. This is not a hardware- or topology-matched training
comparison.

## What the comparison says about executor design

The measurements reinforce four priorities for remote execution:

1. Keep bounded capacity warm by resource profile for the duration of a graph
   or related run series. Provisioning an instance per work unit would move in
   the wrong direction for short nodes.
2. Reduce or overlap per-run local API/broker startup, controller submission,
   and native-agent startup when capacity already exists. The remote SkyPilot
   jobs controller persisted across warm runs. The chain's first callable
   begins after 0.332 seconds locally but 60.433 seconds on AWS; this boundary
   includes snapshot work and multiple lifecycle stages rather than isolating
   any one startup phase.
3. Make the steady-state event and artifact path cheaper. Matching peak
   callable concurrency still leaves roughly eight times the median handoff
   latency and a 56-times larger wide-join gap on AWS.
4. Treat output materialization as a first-class scalability concern. Warm
   fan-out result retrieval averages 0.029 seconds from the local disk
   workspace and 26.402 seconds from the AWS S3 workspace.

An AWS-specific executor could use native instance lifecycle and placement
controls, but these results do not support provisioning in lockstep with each
work unit. The useful unit of provisioning is reusable capacity for a resource
profile—possibly autoscaled from graph lookahead—with logical work units
scheduled onto that capacity. Hard-isolated multi-slot workers, a long-lived
per-worker agent within a bounded executor session, batched metadata
operations, bounded parallel dependency prefetch, and content-addressed
environment reuse are the likely path toward local-like steady-state behavior.
Each optimization still needs an isolated benchmark before assigning it a
saved-time estimate.

## Method, integrity, and limitations

| Component | Local value |
| --- | --- |
| Misen revision | `50ebe5d309e30eb9af0b03ce5054329eb9bde490` |
| Test-project base | `priorcomputers/emergent-geometry` v5, `440d009fb9082a71371c79b675861d9919c73eed` plus benchmark adaptations |
| Executor/workspace | Explicit `LocalExecutor` and fresh `DiskWorkspace` |
| Host | Intel Core i7-10700F; 16 available logical CPUs; 46.9 GiB RAM |
| Primary admission budget | 12 logical CPUs, 32 GiB memory, one GPU |
| GPU | NVIDIA GeForce RTX 3060, 12 GiB class |
| Runtime | Python 3.13.1; PyTorch 2.12.0+cu126; CUDA 12.6 |

CPU and training phases used separate fresh workspaces and environment stores.
All 574 local task hashes were unique and disjoint from the 470 AWS task
hashes. Every run used snapshot `KSKNZ5YVEHDYO`, every durable job state was
`done`, and all 574 job logs contained a start and finish marker with no
failure, traceback, OOM, or CUDA error marker. The benchmark left no worker or
scheduler process running.

The local primary memory value is an admission budget rather than operating
system enforcement, and GPU assignment is cooperative. AWS uses a private S3
workspace, while local uses disk; result-fetch and dependency/join comparisons
therefore include the storage difference. The local scheduler object persists
across CPU runs and is constructed before the timed call, while every attached
AWS run starts its isolated local API/broker and fresh native agents inside the
timed call. Local workers used Python 3.13.1; AWS workers used Python 3.13.15.
Synthetic task timestamps on AWS also assume adequate cross-host clock
synchronization. Sample counts are small, so all ratios are descriptive.

Private evidence retains the per-run JSON, analyzer output, workspaces, and
task logs. The offline analyzer calculates timing statistics, branch-slot
summaries, selected integrity checks, CUDA-result proof, and the comparisons
above without executing tasks or contacting AWS; it writes its JSON summary.
A separate independent audit reconstructed graph topology, validated resource
assignments, and scanned the retained logs.
