# Four-vector sum-table lookup: standalone candidate

The old 2 KiB sum-table candidate waited immediately after each vector gather. The standalone decoder probe now has mode 3, which issues up to four independent sum-table gathers, synchronizes, then consumes their results. Input and final-output copies and 64 repeated iterations use the same four-vector block organization for all modes; volatile input/output accesses prevent benchmark elimination. Temporary gather results are separate from final output.

The gather producer and result-copy loops are explicitly unrolled by four. A rolled-loop prototype returned incorrect data and was not accepted. Its cause has not been isolated; the result is not described as a compiler or hardware defect. Only the passing loop form is retained.

All 65,536 mul1 states match the portable FP16 decoder bit for bit in modes 0, 1, 2, and 3. Additional mode-3 tests for tails of one, two, and three vectors also pass. The existing scalar tile, HVX tile, and standalone matrix checks pass.

Same-run component measurements on the phone:

| Mode | Method | Cycles |
|---|---|---:|
| 0 | Old arithmetic reference | 4,983,629 |
| 1 | Immediate-wait sum lookup | 6,437,522 |
| 2 | Current packed integer rounding | 4,258,164 |
| 3 | Four-vector queued sum lookup | 3,286,040 |

The queued path is approximately 23% shorter than packed rounding in this standalone experiment. These results should not be compared directly with older probes using different copy layouts. They are not model throughput, do not demonstrate 15 tokens/s, and do not include runtime scatter layout, four-worker contention, HMX scheduling, or table/scratch integration.

The production matrix kernel still selects packed arithmetic; no table has been allocated or enabled by this change. A future runtime experiment must preserve scratch bounds, scatter completion before HMX reads, exact weight decoding, all model/API error gates, and end-to-end measurement before adoption. Serialized payload remains 4.805952430792548 bpw and KV remains FP16.
