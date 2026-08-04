# Audit Corrections

Correction time: `2026-08-04T13:20:39.3901135+09:00`

Independent review returned `AUDIT_FAIL` for research-governance consistency. The earlier experiment count treated `HIST-001` through `HIST-007` as new experiments even though they were imported records of work completed before the protocol began. This caused an incorrect total of 24 and an invalid transition to `STOP_NO_EDGE`.

The corrected counts are seven imported pre-protocol experiments, 17 protocol-period new experiments, a limit of 24 protocol-period experiments, and seven remaining slots. Imported records do not count toward the new-experiment limit.

The existing trade calculations, P&L, PF, drawdown, Fold results, costs, rejection decisions and leakage findings remain valid. The correction changes governance metadata only.

The historical preregistration evidence for the 17 protocol-period experiments is marked `INCOMPLETE`. No past timestamp was inferred, fabricated or appended. The final batch conditions were fixed before result generation, but the old JSONL does not provide the strict timestamp-and-hash evidence now required.

The previous `STOP_NO_EDGE` state is preserved as a retracted state-history event. It was cancelled because its experiment-limit premise was false, not because any strategy result improved.

Future experiments use separate hash-linked preregistration and result ledgers. Execution is prohibited until a complete preregistration record has been saved and its script hash verified.

## Second audit correction

Correction time: `2026-08-04T13:46:00.6566918+09:00`

The later `STOP_NO_EDGE` transition at `2026-08-04T13:25:29.6556398+09:00` was also invalid. It relied on an early-stop condition for exhaustion of rationally distinct hypotheses that was not present in the active protocol. That state remains in history but is reclassified as `RETRACTED_BY_AUDIT` with reason `EARLY_STOP_CONDITION_NOT_DEFINED_IN_PROTOCOL`.

The active state returns to `HISTORICAL_RESEARCH`: seven imported pre-protocol experiments, 18 protocol-period experiments, limit 24, and six remaining. No result calculation or rejection decision was changed. No early-stop clause has been added retrospectively. Any such clause belongs only in a future protocol version after the present 24-experiment protocol ends.

## Correction completion

Completion time: `2026-08-04T13:55:24.0164489+09:00`

The six remaining experiments were individually preregistered with actual timestamps, canonical preregistration SHA-256, script SHA-256 and shared-engine dependency SHA-256 before execution. Every execution hash matched and every result was written separately. All six failed their primary acceptance gates.

The corrected count is now seven imported experiments and 24 protocol-period new experiments, with zero remaining. The state is therefore legitimately `STOP_NO_EDGE` under the active protocol's explicit experiment-count limit. Both earlier invalid stop transitions remain visible as retracted history.
