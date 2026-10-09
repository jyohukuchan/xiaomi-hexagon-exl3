# Actual 8,192-token depth: initial execution and speed check

With runtime commit `250465d`, the phone completed this isolated benchmark after the API server was stopped:

```sh
runtime/bin/llama-bench -m models/index-translate-2b-exl3-v2.hxgguf \
  -p 0 -n 32 -d 8192 -b 512 -ub 128 -t 8 -ngl 99 \
  -ctk f16 -ctv f16 -r 1 --no-warmup --progress -o json
```

The pinned benchmark processes 8,192 random prompt tokens before starting its generation timer. It then places 32 further randomly chosen tokens at positions following the retained sequence, synchronizing after each decode. Context allocation is based on depth plus prompt plus generation, not merely an empty 8k capacity. The prefilling and context-state snapshot are outside the timed generation interval.

The executable exited successfully. Its JSON reports depth 8,192, generation 32, batch 512, microbatch 128, eight CPU threads, 99 offloaded layers, and FP16 K/V. The 32 timed steps took 9,576.85 ms, or **3.341391 tokens/s**. The short, real-translation CLI run with the same runtime separately measured 3.44 tokens/s; the prompts, batch sizes, and timing conventions differ, so these are not a controlled speed comparison.

This is one execution/performance sample, not sampled text generation: the benchmark chooses input tokens randomly rather than from logits. It does not check all logits for finiteness, prove long-translation quality, establish sustained thermal performance, or provide a per-operator CPU-fallback audit at this depth. One sample with reported standard deviation zero is not evidence of stable throughput.

The API server was restarted after completion. The 15 tokens/s gate, broader actual-long-context numerical/text validation, 32k checks, and sustained tests remain unfinished. KV is still FP16 and serialized weight payload remains 4.805952430792548 bpw.
