# SkyPilot graph execution on AWS — 2026-09-06

> Historical benchmark: these runs created fresh native agents for each graph.
> The current explicit-session fleet is measured in the
> [September 7 follow-up](skypilot-session-aws-2026-09-07.md).

Status: complete. All ten measurements and both namespaces' scoped cleanup
are verified. In total, 470 logical work units completed using 17 native
allocations, with no initial result-cache hits. Training infrastructure
teardown required a scoped forced GPU termination, documented below; it was
not fully automatic. Final billed cost is unmeasured.

The eight completed CPU runs executed 334 logical work units successfully using
11 native SkyPilot job allocations. A warm two-task smoke took a median
76.390 seconds. A 20-task serial chain took 146.564 seconds for 20.002 seconds
of synthetic useful work. A 102-task fan-out/join took about 303–304 seconds
with two warm workers.

The heterogeneous 68-work-unit training graph, including five CUDA training
tasks, took 409.975 seconds cold and 286.050 seconds warm across three native
allocations per run. All ten training results verified execution on a Tesla T4
and exposed the expected checkpoint keys.

Worker and environment reuse worked, and native job submission was amortized
across many tasks. However, this implementation still has substantial fixed
startup and shutdown costs, roughly 2.6-second typical same-worker task gaps,
and a 17–18-second gap before the 100-parent join. These measurements support
the allocation-based architecture, but do not show low-latency execution for
one-second tasks.

See [SkyPilot usage](../skypilot.md) and the
[graph-execution design](../design_skypilot_graph_execution.md) for the API and
architecture. The [local comparison](local-vs-skypilot-2026-09-06.md) reruns
the same graph shapes with `LocalExecutor`. Measurements below describe the
exact revision tested, not all features proposed in the design.

## Revisions and environment

| Component | Tested value |
| --- | --- |
| Misen graph executor | `50ebe5d309e30eb9af0b03ce5054329eb9bde490` |
| Misen branch | `codex/skypilot-pools` |
| Historical per-work-unit executor | `e14fac391f88a4e462a26ad88ec8dc4f37d171b1` |
| Test project | `priorcomputers/emergent-geometry`, branch `v5` |
| Test-project base revision | `440d009fb9082a71371c79b675861d9919c73eed` |
| SkyPilot | `skypilot-nightly[aws]==1.0.0.dev20260905` |
| Worker Python, all completed CPU runs | `3.13.15` |
| Cloud | AWS `us-east-1`, on-demand, single-node workers |
| Completed CPU measurement window | 2026-09-06, 21:52:06–22:23:23 UTC |
| Cold-training measurement window | 2026-09-06, 22:36:24–22:43:24 UTC |
| Warm-training measurement window | 2026-09-06, 22:44:10–22:49:05 UTC |
| Historical smoke measurements | 2026-09-05 |

The test checkout includes local adaptations to current Misen `FileMap` and
accelerator-resource APIs, plus the benchmark workloads and driver. It is not
an unmodified upstream `v5` checkout. Project dependencies, including
CUDA-enabled PyTorch, were retained for the CPU tests.

| Workload | Graph | Worker capacity |
| --- | --- | --- |
| Original smoke | 2 independent work units | 1 × c6i.large: 2 vCPU, 4 GiB RAM |
| Serial chain | 20 work units, 19 edges | 1 × c6i.large |
| Fan-out/join | Root, 100 branches, join; 102 work units, 200 edges | 2 × c6i.large |
| Training sweep | 68 work units, 108 edges; 5 GPU work units | 2 × m6i.xlarge CPU workers and 1 × g4dn.xlarge GPU worker |

A separate m6i.xlarge SkyPilot jobs controller, with 16 GiB RAM, was used for
the CPU suite. The training-only rerun uses an m6i.2xlarge controller with
32 GiB RAM following the pool-capacity issue described below. Worker and
controller root disks are configured at 50 GiB. The m6i.xlarge workers have
4 vCPU and 16 GiB RAM; the g4dn.xlarge has 4 vCPU, 16 GiB RAM, and one T4 with
16 GiB GPU memory. All original-namespace instances were terminated before
the training-only rerun; the test plan permits at most three workers plus
the controller simultaneously.

## Method and timing boundaries

Every run uses changed task inputs and checks that initially cached work units
are zero. All completed CPU runs used snapshot `KSKNZ5YVEHDYO` and the same
worker environment path fingerprint, `d023aff89149efbf`. Both training
manifests and all of their agent/task bootstrap commands target the same
snapshot. The training series' fresh namespace and different workers are not
a continuation of the original CPU workers' warm state. All 470 task hashes
across the ten completed runs are unique; cold and warm training each contain
68 distinct hashes, with no overlap and different input seeds.

The executor borrows explicit pool capacity. Each attached run starts a fresh
isolated local API/broker lifecycle and fresh native worker-agent jobs. Each
agent executes one child process at a time; successive logical tasks reuse
the VM and installed environment, not the Python interpreter. The controller
and pool workers persist between runs. The driver, rather than Misen's normal
borrowed-capacity cleanup, owns eventual pool/controller teardown.

The workspace is private S3 storage. Coordination uses known-key workspace
mailboxes, with a configured 0.2-second polling interval, not SSH task
dispatch. Submitter credentials are loaded from the default AWS profile into
process environment variables; workers use an instance role for S3 access.
Credentials and cloud/account identifiers are excluded from this report.

| Metric | Definition |
| --- | --- |
| Blocking submit | Monotonic wall time around `executor.submit(..., blocking=True)`, including snapshot creation, local API startup/shutdown, attached coordination, native submissions, task execution, and run cleanup |
| Result fetch | Separate post-submit durability checks and output retrieval; not included in blocking submit |
| Callable time | The synthetic task's returned monotonic body duration; excludes child startup, imports, input materialization, and result publication |
| Task span | Earliest callable start to latest callable end, using worker wall timestamps |
| Chain handoff | Child callable start minus its parent's callable end, on the same worker |
| Branch handoff | Gap between consecutive branch callables on the same worker, excluding root-to-branch and branch-to-join transitions |
| Join gap | Join callable start minus the latest branch callable end |
| Native allocation | An accepted `allocation-*.json` record with a durable SkyPilot request/native identity; not a manifest slot, VM count, or child-process count |

The chain and fan-out bodies deliberately sleep for one second. Their
"useful work" is synthetic elapsed time, not CPU utilization or a compute
throughput benchmark. The original smoke returns worker information but does
not record a comparable monotonic callable duration.

Pool apply is outside the measured submission. Its return does not establish
worker readiness, so the first run includes remaining worker/environment
startup. "Warm" means worker/environment reuse, not an already running native
agent or local API service. The first fan-out adds a second worker and is
therefore partly cold.

Protocol archiving and read-only cloud inventory happen after submission and
result fetching. Their encompassing stage duration is not the headline time.
No extra SkyPilot status/log calls were made during timed graph runs. Periodic
read-only S3 run-state checks were made during synthetic and training runs,
roughly once per minute during fan-out, with a check near chain completion.
These diagnostic checks did not mutate workloads, but this was not a
completely unobserved experiment.

## Verified CPU results

All eight runs finished with every logical job `done`, durable outputs present,
zero initial result-cache hits, and no recorded run-cleanup errors. Native
agents becoming `CANCELLED` during successful cleanup is intentional and does
not mean logical task failure. The original namespace's final pool/controller
and storage cleanup, including the separate training namespace, is verified
below. Infrastructure-cleanup intervention is distinct from logical run
cleanup.

| Run | Work units | Native allocations | Blocking submit, s | Result fetch, s |
| --- | ---: | ---: | ---: | ---: |
| Smoke, cold | 2 | 1 | 178.512 | 0.650 |
| Smoke, warm 1 | 2 | 1 | 79.425 | 0.475 |
| Smoke, warm 2 | 2 | 1 | 76.390 | 0.431 |
| Smoke, warm 3 | 2 | 1 | 75.749 | 0.455 |
| Chain, 20 | 20 | 1 | 146.564 | 5.137 |
| Fan-out, first two-worker run | 102 | 2 | 333.199 | 26.666 |
| Fan-out, warm 1 | 102 | 2 | 304.405 | 26.677 |
| Fan-out, warm 2 | 102 | 2 | 302.961 | 26.128 |

Warm smoke: median 76.390 seconds, mean 77.188 seconds, three samples. Warm
fan-out: mean 303.683 seconds, two samples, 8.9% below the first two-worker
run. These are descriptive comparisons, not statistical performance claims.

### Callable execution and handoffs

| Run | Sum of callable time, s | Task span, s | Handoff median, s | Handoff p95, s | Handoff max, s | Join gap, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chain, 20 | 20.002 | 71.726 | 2.627 | 3.352 | 3.983 | — |
| Fan-out, first | 102.012 | 246.138 | 2.553 | 3.892 | 4.633 | 17.504 |
| Fan-out, warm 1 | 102.013 | 213.471 | 2.670 | 4.149 | 5.681 | 17.633 |
| Fan-out, warm 2 | 102.011 | 219.710 | 2.658 | 4.923 | 6.796 | 18.266 |

Chain statistics cover 19 dependency handoffs. Fan-out statistics cover
98 same-worker branch-to-branch gaps per run; branches are independent, so
these are not dependency latencies. Percentiles use linear interpolation.
Root-to-branch delays include waiting for the two available agents and should
not be compared directly with serial-chain handoffs.

The chain's first callable started 60.433 seconds after submission began.
Its 19 handoffs consumed 51.725 seconds; its final callable ended at
132.160 seconds, followed by 14.406 seconds until blocking submission returned.
Thus 20 seconds of task bodies occupied about 72 seconds between the first
and last callable, inside a 147-second end-to-end submission.

Both fan-out workers executed callables concurrently, with a measured maximum
of two simultaneous callables. This does not mean the available CPUs were
continuously busy. All 102 tasks used distinct host/PID pairs per run; the
chain similarly used 20 distinct child PIDs on one worker.

### Worker reuse and capacity ramp

Worker A is the original small-pool host. Worker B is the additional host
introduced for fan-out. Private hostnames are intentionally omitted.

| Fan-out run | Tasks on A / B | A's first callable after submission, s | B's first callable after submission, s | B's start lag behind A, s |
| --- | ---: | ---: | ---: | ---: |
| First | 62 / 40 | 59.159 | 143.089 | 83.931 |
| Warm 1 | 51 / 51 | 65.912 | 75.361 | 9.449 |
| Warm 2 | 52 / 50 | 58.480 | 78.220 | 19.740 |

All smoke and chain tasks ran on A. Later fan-out runs reused A and B, with
matching Python executable paths and snapshot identities. The first fan-out's
late second worker explains part of its longer task span and uneven task
distribution. Warm worker availability does not eliminate staggered native
agent submission and startup.

## Remaining overhead and implications

1. Fixed per-run lifecycle cost is still large. In warm smoke runs 1 and 2,
   coarse object timestamps place roughly 44 seconds between manifest
   persistence and the accepted native-job record, followed by 12–14 seconds
   until the first assignment. Final result to submit return adds roughly
   13–14 seconds. These are boundary intervals, not isolated measurements of
   SkyPilot SDK submission or a particular internal operation.
2. Reusing an environment does not reuse a process. The approximately
   2.6-second median handoffs include result commit, polling/queueing, a fresh
   child interpreter, environment checks, imports, and input retrieval.
   Installed dependencies persist, but Python imports and bootstrap still
   repeat. Reducing cloud provisioning alone cannot remove this cost.
3. Wide joins have measurable artifact/dependency overhead. The 100-parent
   join began 17.5–18.3 seconds after its final parent ended. This includes
   materializing dependencies and bootstrap; it is not pure scheduling delay.
4. Post-submit output retrieval is separate and scales with exposed results:
   about 5.1 seconds for 20 outputs and 26.1–26.7 seconds for 102 outputs,
   including durability checks. The small synthetic result values make this
   evidence about per-object overhead, not bulk-transfer bandwidth.
5. Graph size no longer determines native submission count directly: 20 tasks
   shared one native allocation, and 102 tasks shared two. The next latency
   targets suggested by these data are per-run agent startup, fresh-process
   execution, and workspace/dependency I/O. This benchmark does not measure
   the benefit of any unimplemented optimization.

Three concrete, unmeasured optimization hypotheses follow from the evidence
and inspection of the tested scheduler:

- A verified-environment fast path could skip redundant payload
  bootstrap/materialization after an agent verifies the installed snapshot,
  while retaining a fresh child process, its guard, execution claims, and
  resource isolation.
- Bounded parallel prefetch of already committed dependency artifacts could
  shorten wide joins, while retaining integrity checks and atomic publication
  semantics.
- Bounded profile lookahead could start a GPU agent while its CPU prerequisites
  run. Currently, native agents launch only when that resource profile has a
  ready task; pre-staging agent commands in the manifest does not launch them.
  Prewarming could overlap agent startup while keeping payload execution
  dependency-ready-only, at the cost of potentially idle billable capacity and
  additional cancellation handling.

None of these changes was implemented or benchmarked here; their saved time
and correctness need separate tests.

Object-store `last_modified` timestamps have coarse resolution and describe
the latest persisted record. In particular, an allocation timestamp is not
the original SDK launch-call start: warm training's GPU native-ID record was
updated after its first execution-start marker. No exact provisioning,
installation, or SDK cost split is claimed. Same-worker gaps avoid
inter-worker clock skew; cross-host
spans, join gaps, and driver-to-worker offsets assume sufficiently aligned
wall clocks. Monotonic submission and callable durations do not need that
assumption.

## Historical smoke comparison — indicative only

The previous pooled implementation launched a managed job for each work unit.
Its 2026-09-05 fixed-code rerun used the same worker/controller instance shapes
and SkyPilot version. It measured one cold/warm pair:

| Smoke implementation | Cold, s | Warm, s | Native jobs per two-task run |
| --- | ---: | ---: | ---: |
| Prior per-work-unit managed jobs | 216.500 | 113.193 | 2 |
| Graph executor | 178.512 | 76.390 median, n=3 | 1 |

The descriptive reduction is 17.5% cold and 32.5% warm. This is not a
controlled exact A/B comparison: the old harness times a CLI subprocess,
including CLI parsing and initial Python imports, whereas the new driver
times blocking submission from an already running Python process. Both
include their API lifecycle and snapshot work. Source revisions, measurement
dates, and locally adapted project snapshots differ, and sample counts are
small. Old worker logs measured 0.45–0.61 seconds per smoke function; the new
smoke does not record equivalent callable durations.

The previous executor rejected pooled graphs with pending dependencies to
avoid pool deadlock. There is therefore no matched-capacity pooled baseline
for the chain or fan-out, and no measured speedup over that baseline is
claimed for dependent graphs.

## Evidence and artifact accounting

Across the ten completed runs, JSON summaries contain 344 exported output
values: all 334 CPU-suite outputs and ten selected training summaries. All
470 work units were checked for durable outputs, but the other training
outputs were not individually deserialized into the summary. The ten
companion protocol exports contain 2,908 record snapshots, including 470
durable completion records and 17 accepted native-allocation records.

The private local evidence archive retains eight run summaries and eight
companion protocol exports for the completed CPU phases. The summaries
contain 334 exported output values. The protocol exports contain 2,064 JSON
record snapshots, including 334 durable logical completion records and
11 accepted native-allocation records. These counts are not counts of S3
requests, all workspace objects, or transferred bytes: mutable worker/run
records are exported as their latest state, not a complete event history.

Before deleting the two workspace buckets, the driver archived their objects:

| Archive | Objects | Bytes |
| --- | ---: | ---: |
| Original CPU / failed-pool-setup namespace | 6,114 | 6,710,223 |
| Training namespace | 2,665 | 17,946,651 |
| Total | 8,779 | 24,656,874 |

A separate offline audit verified every archived object against its indexed
size and SHA-256 digest, with no missing objects, duplicate keys, or mismatches.
These full-bucket archives are broader than the protocol-record snapshots and
preserve raw artifacts without requiring their deserialization.

Run labels are `smoke-cold`, `smoke-warm-1`, `smoke-warm-2`, `smoke-warm-3`,
`chain-20`, `fanout-first`, `fanout-warm-1`, `fanout-warm-2`, `training-cold`,
and `training-warm`. The local drivers are `scripts/aws_graph_benchmark.py`
and `scripts/aws_training_benchmark.py`; the read-only analyzer is
`scripts/analyze_aws_graph_benchmark.py` in the adapted test checkout.
The analyzer consumes JSON evidence only, makes no AWS calls, and does not
automatically unpickle archived outputs. Raw evidence location, cloud
identifiers, and private hostnames are retained in the private handoff rather
than published here.

## Heterogeneous training sweep — verified

The first GPU pool apply failed before any training began, reporting
`Failed to spin up the service` after approximately 67.9 seconds. The
controller reported its maximum pool-service count was already reached
at 1/1: the CPU pool occupied the only slot permitted by the 16 GiB controller
under this SkyPilot build's non-consolidated memory accounting. This was a
pool-setup failure before GPU provisioning, not a failed training measurement.

The training-only rerun uses a fresh namespace and an m6i.2xlarge controller
with 32 GiB RAM, retaining the same two CPU-worker/one-GPU-worker limits. GPU
pool apply then succeeded, and `training-cold` began at 22:36:24 UTC on
2026-09-06. The change is benchmark controller sizing, not a Misen
production-code fix.

The reduced sweep keeps all five variants and downstream analysis, with eight
training steps, batch size eight, sequence length sixteen, and two checkpoints
per variant.
Each graph has 63 CPU work units and five CUDA-required work units. GPU tasks
must fail rather than silently fall back to CPU.

| Run | Work units | Native allocations | Blocking submit, s | Result fetch, s | Sum of five training loops, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training, cold | 68 | 3 | 409.975 | 9.297 | 2.496 |
| Training, warm | 68 | 3 | 286.050 | 8.966 | 2.489 |

Each run completed all 68 logical work units with durable outputs,
108 dependency edges, zero initially cached work units, and no recorded
run-cleanup errors. Each has three accepted native allocation records with
durable request/native identities. Manifest, assignment, and worker-agent
profiles agree: 63 work units used CPU capacity, distributed 34/29 in the cold
run and 30/33 in the warm run, and all five GPU work units used the GPU agent.
All three agents stopped after each execution; this is not final
pool/controller teardown. Warm blocking submission was 30.23% shorter than
cold, based on one cold/warm pair.

All ten returned training summaries report `cuda:0`, `Tesla T4`, CUDA `12.6`,
and the same GPU worker hostname across both runs. Each has four metrics rows,
finite loss and accuracy, checkpoint keys `[4, 8]`, and final checkpoint step
`8`. Individual training-loop durations range from 0.492 to 0.504 seconds.
The export verifies checkpoint references/keys, not a separate reload of each
checkpoint file.

The 2.496/2.489-second sums measure synchronized training loops, including
their checkpoint saves. They are **not total useful compute for the graph**:
CPU preprocessing/postprocessing, model initialization, imports, environment
setup, and other work are not included. Subtracting these sums from the
end-to-end runtimes would not establish exclusively scheduler overhead.

| Boundary after submission began | Cold, approximate s | Warm, approximate s |
| --- | ---: | ---: |
| First starts on the two CPU agents | 108 / 128 | 55 / 66 |
| First GPU execution start | 237 | 119 |
| Final GPU result committed | 271 | 152 |
| Final CPU result committed | 369 | 246 |
| All worker agents stopped | 371 | 248 |
| Run-state record `done` | 406 | 280 |
| Blocking submission returned | 410 | 286 |

These are coarse object-store marker boundaries, not callable timings.
CPU/postprocessing and run cleanup continue after the training loops finish.
GPU agent execution did not begin alongside the earliest CPU work even in the
warm run, consistent with the ready-profile launch policy discussed above.
No measured saved-time estimate is assigned to prewarming.

Each training run adds five exported training summaries and 422 protocol
record snapshots, including 68 durable completion records and three accepted
native allocation records. Actual training Python executable paths and child
PIDs were not included in the training summaries; matching bootstrap snapshot
targets are not a substitute for that missing runtime evidence.

## Bounds, cleanup, and cost

Each graph is configured with a 15-minute run bound, a separate 900-second
setup timeout, and a 60-second shutdown timeout. The original outer benchmark
driver has a 70-minute bound; the new training-only driver has a 35-minute
bound. These are execution controls, not a cloud billing cap. The target
budget is below US$10, not a hard guarantee or verified final cost.

Per-run logical cleanup success does not establish infrastructure teardown.
The original CPU/failed-pool-setup namespace has completed teardown:

- `termination.json`, recorded at 22:30:55 UTC, verifies no live test-owned
  instances or remaining root volumes. Its five exact instance identities
  match the ownership inventory: two c6i.large workers, two m6i.xlarge workers
  created for the first training setup, and the m6i.xlarge controller. Five
  associated root volumes are absent.
- `cleanup.json`, recorded at 22:35:47 UTC, verifies cloud-resource removal,
  bucket deletion, and no remaining local namespace API/broker processes,
  with no SDK cleanup error. The archive was completed before bucket deletion.
- An independent offline comparison confirms that all ten recorded preexisting
  instance states are unchanged. Other namespaces were outside teardown scope.

These findings audit the recorded cleanup evidence; the audit itself made no
AWS calls. The terminated instances and deleted disks/bucket objects cannot
serve the original workloads; the verified local archive is retained.

Cleanup of the training namespace began at 22:49:47 UTC and completed at
22:57:39 UTC. Infrastructure teardown required intervention:

- The initial cleanup stopped at 22:53:26 UTC with a `RuntimeError` because
  its 120-second termination wait expired. The two CPU instances, controller,
  and their root volumes were gone, but the GPU instance had remained
  `shutting-down` since 22:50:26 UTC. All measured workloads had already passed.
- At 22:54:51 UTC, the operator rechecked the exact GPU identity and ownership
  tags, absence from the preexisting-instance baseline, terminal training runs,
  and workspace bucket ownership. Only that test-owned GPU received
  `Force=True, SkipOsShutdown=True` termination. This follows
  [AWS's documented stuck-termination procedure](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/TroubleshootingInstancesShuttingDown.html).
  The private evidence retains the scoped request and response. Acceptance of
  this request alone is not proof that termination finished.
- Cleanup resumed at 22:55:05 UTC. The 22:55:16 UTC inventory shows all four
  training-namespace instances terminated, and the subsequent termination
  record verifies no remaining root volumes. At 22:55:26 UTC, retrying the
  already removed SkyPilot controller produced an `ExecutionError`; the
  ownership-scoped fallback found no remaining instances and made no further
  termination calls. Archival began at 22:55:27 UTC and finished at
  22:57:35 UTC after 127.475 seconds. The SDK retry error must not be mistaken
  for flawless automatic teardown.
- The final independent AWS proof at 22:56:09 UTC confirms all four exact
  training instances terminated, all four root volumes absent, and all four
  associated network interfaces absent. Its instance/volume identities match
  the recorded ownership inventory.
- `cleanup.json`, recorded at 22:57:39 UTC, verifies bucket and cloud-resource
  removal and no remaining local namespace processes. All 15 recorded
  preexisting instance states match the training baseline, including the five
  already terminated instances from the original namespace. The record
  retains `sdk_error_type="ExecutionError"` for the controller retry above.
- Follow-up read-only checks at about 22:58 UTC independently confirmed both
  workspace buckets return 404 and neither namespace has local API/broker
  processes remaining.

Across both namespaces, nine distinct test-owned instances are terminated and
their nine root volumes are absent. Both workspace buckets were deleted after
archival; the verified local copies remain. Cloud storage removed by cleanup
is not recoverable from those deleted disks/buckets. The cleanup audit itself
was read-only and did not contact AWS; it checked the retained API evidence,
ownership/baseline identities, state comparisons, and every archive hash.

Final billed cost is **unmeasured**. The target budget above is not an
actual-cost result, and this report makes no claim about charges outside the
explicit benchmark resources.

## Scope and limitations

This tests attached execution in one AWS region, fixed small pools, and
single-node allocations. It does not validate multi-node/Dask execution,
detached coordination, preemption, retry, dynamic worker replacement, fault
recovery, or behavior at substantially larger scale. CPU synthetic bodies are
sleep tasks, not representative CPU kernels. The training sweep is intended
to exercise heterogeneous dependencies and artifact movement, not evaluate
model quality. Few warm repetitions and mixed timing boundaries limit
comparisons; no subsecond scheduler-latency or production-readiness claim is
supported by this benchmark.
