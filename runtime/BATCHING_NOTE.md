# Why the first batching sweep was discarded

`asr_core.decode()` calls `model.transcribe(paths, batch_size=1)` — the batch
size is fixed at 1 inside the shared path. Handing it more paths therefore does
NOT batch the model; it only amortises per-call setup.

Measured consequence: the adapter, whose backend loops over paths, showed
8,986 -> 8,919 ms/clip going from "batch" 1 to 4 (1.01x, i.e. nothing), while
the NeMo CTC model showed 79.2 -> 40.5 ms/clip (1.96x) purely from amortising
NeMo's per-call dataloader construction.

Comparing those two numbers would have been dishonest: one is overhead
amortisation, the other would have been true batching. A real batching
comparison must call each model's own batched inference path (for the adapter,
one `generate()` over a padded batch; for NeMo, `transcribe(batch_size=N)`),
and must do so for every model or for none.

The 8.1x figure quoted elsewhere for the adapter came from a genuinely batched
run (batch 16 through the pipeline's own batched decoder) and is a
within-model THROUGHPUT result, not a latency result and not cross-model.
