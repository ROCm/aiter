# Published implementation tickets

The user approved the four-ticket breakdown on 2026-10-10. These are the sole current implementation tickets, initially published as `ready-for-agent`. Current state is maintained only in the integration worktree and its [map](../map.md):

| Ticket | Blocked by | Status |
| --- | --- | --- |
| [01: Safe tuning lifecycle](01-safe-tuning-lifecycle.md) | None | resolved |
| [02: Complete default staged search](02-default-staged-search.md) | 01 | claimed |
| [03: Configured execution and AOT](03-configured-execution-and-aot.md) | 02 | ready-for-agent |
| [04: Calibration and acceptance](04-calibration-and-acceptance.md) | 03 | ready-for-agent |

Work the frontier in order: 01→02→03→04. Ticket 01 is resolved with recorded acceptance and ticket 02 is claimed; tickets 03–04 remain blocked. See the ticket Answers and map for implementation and merge commits. After publication, the user authorized the pre-implementation reviewer corrections; the current tickets and [specification](../spec.md) now include those clarifications, and spec is `ready-for-agent` with its pause removed. The [work map](../map.md) tracks current state, and [reviewer decisions](../reviewer-comments.md) record the dispositions. The four-ticket order and staged algorithm remain unchanged.

The seven/ten drafts and approved four-ticket snapshot are [archived](../archive/README.md). Archived copies are historical references and cannot be claimed.
