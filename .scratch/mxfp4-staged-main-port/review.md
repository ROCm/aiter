# Final planning consistency review

Status: ready-for-human

Reviewed on 2026-10-10 against user decisions Q1–Q14 and target main `882ea79dd7cc7091f859a5bedac914523ac77bbd`. Two independent read-only investigations reviewed the specification, implementation plan, acceptance matrix and seven tickets. No migration code or GPU execution was performed.

Those reviews reported no unresolved substantive contract, evidence, scope or dependency conflict. They confirmed consistency with the decisions then recorded, including M_est-pruned search, layout-v2-only scope, grouping, settings, gate, CSV/device/Pipe contracts and calibration. This historical report was not proof that all deployment-configuration and success-coverage risks had been resolved; the later reviewer comments below correct that scope.

One wording issue was corrected: worker-returned Python shape exceptions are failed shape results; the clean-EOF distinction applies specifically to IPC receive/protocol errors, while unexpected IPC errors fail the supervisor. The complete process/device contract is now also stated in the spec.

Local verification checked all planning Markdown links and trailing whitespace, all 14 decision records, seven tickets and an ordered acyclic dependency graph. These checks validate the planning artifacts, not the unimplemented port. The source branch's old tests/calibration are explicitly historical evidence.

At the time of this review the design was awaiting the user's final shared-design confirmation. The user subsequently requested `to-spec` on 2026-10-10, publishing the formal specification as `ready-for-agent`. Existing implementation tickets remain prior draft breakdowns for ticket finalization, without being claimed or implemented.

After that review, the user requested a deeper audit of Cartesian partitioning and full fMoE execution. The [pipeline grouping audit](pipeline-grouping-audit.md) confirms the static compatibility partition but does not establish exact performance separability; it also identifies the need to confirm production's actual input dtype/prequant path in addition to names. Acceptance and ticket 06 now make that check explicit. This follow-up has not approved a new merge/split policy or removed normal end-to-end validation.

## Formal specification publication

The user's `to-spec` request on 2026-10-10 consolidated the confirmed design, audit and completed cross experiment into the prescribed seven-section specification, with 48 user stories and `Status: ready-for-agent`. Implementation decisions describe behavior/interfaces without concrete source paths or code snippets; supporting evidence remains linked separately.

Two independent read-only checks found no substantive omissions or contradictions in the published spec. They verified the unchanged candidate/search/failure/deployment contracts, existing test seams, 17+10 calibration matrix, no hard performance target, same-card sequential GPU tests, and bounded experiment conclusions. Link/whitespace/template checks passed. The earlier decision text is retained in [design-decisions.md](design-decisions.md), explicitly subordinate to the published spec. Prior draft implementation tickets remain unclaimed and require ticket finalization; no code or GPU work was started by this publication.

## Reviewer corrections and current disposition

The later reviewer found gaps those consistency checks missed: configuration identity was not enforceable from the CSV and lacked a specified record/checker; standalone plan visibility advice could select the wrong physical GPU; reference/all-stage failure could lose a valid prune result under the recorded skip policy. The earlier reviews checked that the text agreed with Q1–Q14 and linked artifacts, but did not sufficiently challenge these deployment and outcome-coverage assumptions. Their “no substantive conflict” conclusion must not be used as current clearance.

At this review stage, the user had revised skip-on-reference/all-screening-failure to exhaustive normal pair validation within the affected pruned group, with returned pair cache reuse. Reference retry scope, `--mp`, matrix wording, GPU identity instructions and user-story structure were clarified. Seven/ten drafts were archived and the unapproved four-ticket proposal was the current breakdown, with no ready or claimed tickets. Physical machine identifiers remained in acceptance, and spec retained identity/same-card principles. The later publication and pre-implementation corrections below supersede that ticket state.

The user assigned winner-file/settings consistency to the caller and declined additional automatic handling. No sidecar, configuration binding, mismatch rejection or prescribed deployment metadata format is required. The tuner environment entry/snapshot and caller obligation were made explicit; calibration retained the agreed separate outputs/settings. The dispositions are recorded in [reviewer-comments.md](reviewer-comments.md). At that stage spec/plan/acceptance remained `needs-info` during the user's publication/implementation pause, with no open behavior choice. That historical status is superseded below. No GPU or implementation validation was performed in this revision; these dispositions are not a claim of completed-port acceptance.

## Approved ticket publication

On 2026-10-10 the user approved the four-ticket breakdown after two read-only checks of contract coverage and true blocking edges. The sole current set is now [published under issues](issues/README.md), all `ready-for-agent`, with dependencies 01→02→03→04. The checks confirmed explicit staged `--mp` selection and the existing tuner limit environment/invocation snapshot alongside the agreed fallback and caller-owned configuration rules. Prior seven/ten drafts and the approved four-ticket snapshot are archived. Earlier pause/proposal statements above describe historical review states and are superseded for ticket publication by the [work map](map.md). The parent spec was not modified; no ticket was claimed and no implementation or GPU test started through publication.

## Pre-implementation reviewer correction

The next reviewer identified issues the publication check did not resolve: leaving the parent spec paused made the ticket entry point contradictory; all-mode runtime preflight was only implicit outside the staged subsection; and shared changes needed a concrete non-MXFP4 lifecycle regression. Production's per-call output-flag reads also required an explicit stable-environment contract, and caller-owned CSV responsibility needed the same-key default/limit 5 skip-or-overwrite example. The earlier publication check proved ticket copying/link/status consistency, not that all operational ambiguities were resolved.

After the user's explicit instruction to process these comments, spec was restored to ready-for-agent and the current spec/tickets/plan/acceptance were updated with all-mode device preflight, real non-MXFP4 controlled-outcome regressions, worker environment/snapshot/production-path parity, the concrete CSV hazard, 02's independently reviewable internal steps and pre-search full-anchor cost records. The old bare logical-GPU selection advice was corrected. The approved four-ticket chain is retained as a delivery order; some internal AOT repairs could technically precede 02 without changing that whole-ticket gate.

See [second-reviewer dispositions](reviewer-comments.md#second-reviewer-pre-implementation-corrections) for each judgment and applied requirement. These changes were made directly without subagents, implementation or GPU execution. Counts, wall-time estimates, numerical/deployment evidence and behavioral regressions are requirements for the implementation tickets, not completed checks from this documentation revision. The archived approved-draft snapshot remains unchanged.

Documentation verification passed for all 42 feature Markdown files: links/heading anchors, trailing whitespace, the seven-section/46-story spec, four ready tickets and unchanged blocking edges. Comparison with the pre-revision snapshot confirmed unchanged candidate/grouping/screening/fallback sections, scope exclusions, mode table and both calibration tables; archived files are byte-identical. Repository diff formatting checks passed. These checks validate the documentation, not implementation acceptance.
