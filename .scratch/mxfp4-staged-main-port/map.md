# MXFP4 staged main-port work map

Status: claimed

The user approved the four-ticket breakdown on 2026-10-10 after [spec reviewer resolution](reviewer-comments.md). The sole current implementation tickets are published under [issues](issues/README.md); tickets 01–03 are resolved and ticket 04 is claimed, with the blocking edges below. Implementation was authorized by the user; tickets 01–03 are resolved and ticket 04 is active. Target main remains `882ea79dd7cc7091f859a5bedac914523ac77bbd`.

| ID | Ticket | Blocked by | Status |
| --- | --- | --- | --- |
| 01 | [Safe tuning lifecycle](issues/01-safe-tuning-lifecycle.md) | — | resolved |
| 02 | [Complete default staged search](issues/02-default-staged-search.md) | 01 | resolved |
| 03 | [Configured execution and AOT](issues/03-configured-execution-and-aot.md) | 02 | resolved |
| 04 | [Calibration and acceptance](issues/04-calibration-and-acceptance.md) | 03 | claimed |

The approved delivery sequence is 01→02→03→04. Ticket 01 was the initial frontier; later tickets require their recorded blockers to be completed. Some AOT and prune/full parameter repairs in 03 could technically be prepared before 02, but the approved ticket order is retained to coordinate shared edits and verify complete configured staged behavior before closing 03. These are completion gates for whole tickets, not a claim that every internal change has the same code dependency. No partial-ready states or new blocking edges are introduced. Ticket 01 is resolved after integration merge and recorded acceptance. Ticket 02 is resolved after default numerical/production/AOT acceptance; ticket 03 is resolved with configured numerical/production/AOT acceptance; ticket 04 is claimed and is the current frontier.

See [spec.md](spec.md), [plan.md](plan.md), [acceptance.md](acceptance.md) and [facts.md](facts.md) for contracts, execution order, checks and evidence. After ticket publication, the user authorized the pre-implementation reviewer corrections: spec is now `ready-for-agent`, its pause is removed, and current tickets include the clarified device/regression/environment/cost-planning requirements. This map tracks current ticket state; the spec defines behavior. The four-ticket order and staged algorithm remain unchanged.

The [review record](review.md) includes the later corrections and the user's fallback/caller-ownership decisions. Seven/ten drafts and the approved four-ticket snapshot are [archived](archive/README.md); only files under `issues/` are current implementation tickets.

The [crossed-pair experiment](cross-experiment-review.md) is complete and its findings are incorporated in the plan and acceptance. Experiment artifacts are committed separately at `96e2b5545`; no port implementation ticket has been completed by that experiment or this publication.

## Implementation coordination

The sole mutable ticket-state location is this integration worktree: `/root/workspace/tmp_dev/aiter-mxfp4-integration/.scratch/mxfp4-staged-main-port/`. Only the primary coordinator updates ticket status, Answers and this map. Implementer copies are read-only context; implementation branches submit code, behavior documentation and evidence without tracker-state edits. The original source worktree remains unchanged.

Integration branch: `integration/mxfp4-staged-main-port`, fixed base `882ea79dd7cc7091f859a5bedac914523ac77bbd`. Current frontier: ticket 04 (claimed); tickets 01–03 resolved at merges `47cb9fb46d8ea532beea98d6dc7d920475f7ece6`, `00f9c6073ff3d078ee5ead9688f7dc6804a337e6` and `1e901d768`.

## Ticket 01 delivery

Ticket 01 resolved with implementation `3a05e8ce0`, merged as `47cb9fb46d8ea532beea98d6dc7d920475f7ece6`. Its Answer links 55 controlled lifecycle and 55 shared infrastructure checks plus static verification. The early uncontrolled-execution incident remains disclosed in delivery evidence and is excluded from acceptance. Ticket 02 starts from the resulting integration tip.

## Ticket 02 delivery

Ticket 02 resolved at integration merge `00f9c6073ff3d078ee5ead9688f7dc6804a337e6`, with 155 controlled checks and four serialized real GPU tasks (staged winner, exact production, fresh AOT, precompiled-only exact replay) all passing. Its Answer links complete evidence and preserves metric/configuration boundaries. Ticket 03 starts from the resulting tip.

## Ticket 03 delivery

Ticket03 resolved at integration merge `1e901d768`, with 231 controlled cases and 90 serialized GPU tasks passing. Its Answer links configured path/AOT/numerical/clamp evidence and distinguishes tuner-default explicit Situv2 parameters from the omitted API check still required in04. Ticket04 starts from the resulting integration tip; all27 searches, cost planning and final review remain to be performed.
