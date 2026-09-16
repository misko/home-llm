**A local benchmark for Hebbian learning and small language models**

Recommended starting point: an approximately 7.35M-parameter transformer trained from scratch on a fixed TinyStories subset, paired with multi-query associative recall (MQAR) diagnostics. Use a second, small FineWeb experiment to check whether promising results extend to ordinary web text. Borrow Parameter Golf's fixed resources, reproducible evaluation, and compression metric, while setting budgets appropriate to one RTX 4090.

This is an experimental proposal, not a report of completed training. After the user stopped the Qwen server on September 12, 2026, hardware inspection found 22,062 MiB (approximately 21.5 GiB) free on the 24 GB RTX 4090, with ample system RAM. This is a snapshot, not a reservation of GPU memory. Timing examples below are conditional calculations, not measured throughput on this machine. No training workload has been launched for this benchmark.

**What Parameter Golf contributes.** OpenAI's challenge specifies a 16,000,000-byte artifact and ten minutes of training on eight H100 GPUs, with FineWeb validation scored in bits per byte. The original baseline uses nine blocks, width 512, and a 1,024-token vocabulary. A ten-minute run on a single consumer GPU has a different compute budget. The original implementation and its evaluation/data design are useful references, but reproducing its leaderboard score is a separate task. [Official repository](https://github.com/openai/parameter-golf), [baseline code](https://github.com/openai/parameter-golf/blob/main/train_gpt.py).

Adopt fixed data, public model configurations, bounded training budgets, three seeds for final comparisons, and separate reporting of model quality, storage, and runtime. Record compilation and preprocessing time rather than hiding them. Initially compare BF16 models and explicit parameter counts; aggressive quantization would introduce another variable before the learning rules have been understood. Parameter Golf's compressed artifact limit is different from parameter count, training memory, and inference memory.

**Benchmark stages.** The following are proposed experiment sizes and budgets, rather than published benchmark settings or promised completion times.

| Stage | Data/task | Model size | Initial budget | Main question |
|---|---|---|---|---|
| Mechanism diagnostic | MQAR and overwriting variants | Small matched state/parameter budgets | 5–10 minutes per configuration | Can the rule bind, retrieve, retain, and replace associations? |
| Pipeline debug | TinyStories, 10M training tokens | 1.31M parameters | 10-minute time cap | Does the training/evaluation implementation work? |
| Main experiment | TinyStories, fixed 100M-token stream | 7.35M parameters | 15-minute screen; 60-minute main comparison | Does the approach learn useful next-token distributions? |
| Transfer check | Fixed FineWeb subset, initially 100M tokens | 7.35M; optionally 27.27M | 60–120-minute comparison | Does the result persist on less restricted text? |

Use two separately labeled views: loss at equal token exposure and loss at equal wall-clock budget. A run ending at its time cap may not have consumed all 100M tokens. Do not label it a 100M-token result. Longer equal-token runs can have a separate hard safety cap and report incomplete runs explicitly.

TinyStories is intentionally restricted synthetic text, and its original study demonstrates coherent generation with models below 10M parameters. This makes it a good setting for fast learning-rule experiments; it is not evidence of broad knowledge, chat capability, or general reasoning. Use held-out next-token loss as the primary metric, with a fixed set of story prompts for qualitative inspection. [TinyStories paper](https://arxiv.org/abs/2305.07759), [dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

MQAR tests retrieval of several earlier key/value associations at different locations. It is better suited to diagnosing associative memory than judging a few generated stories. Independently vary number of bindings, distractors, query distance, and sequence length. Add overwrite tests with repeated keys and explicitly specify that the latest value wins. Train and test on independently generated associations, not memorized fixed mappings. [Zoology paper](https://arxiv.org/abs/2312.04927), [official implementation](https://github.com/HazyResearch/zoology).

For a FineWeb follow-up, Parameter Golf provides a cached dataset and supports downloading a limited number of shards. Pin the data and tokenizer revision. A locally reduced dataset or changed tokenizer defines a local benchmark; its scores are not official competition results. Keep this stage separate from TinyStories instead of comparing their perplexities or BPB values as though the corpora were equally difficult. [Parameter Golf data documentation](https://github.com/openai/parameter-golf/tree/main/data).

**Proposed transformer baseline.** Use a plain causal decoder with the following configuration:

| Setting | Main baseline |
|---|---|
| Blocks | 8 |
| Width | 256 |
| Attention | 4 query/key/value heads, head dimension 64 |
| MLP | Two bias-free linear layers, hidden width 1,024, GELU |
| Normalization | Two RMSNorms per block and one final RMSNorm |
| Position encoding | RoPE |
| Tokenizer | Fixed 4,096-token lossless BPE with byte fallback |
| Token embeddings/output | Tied |
| Context | 512 tokens |
| Parameters | 7,344,384 under this exact specification |
| Training | BF16, AdamW, causal next-token cross-entropy |
| Initial effective batch | 8,192 tokens per optimizer update; tune only if justified |
| Implementation | PyTorch SDPA, pretokenized data, compiled/eager timings reported |

The count is Vd + 12Ld² + (2L+1)d, with no learned positional embeddings or linear biases. A debug variant with four blocks at width 128 has 1,311,872 parameters; a scale-up variant with eight blocks at width 512 has 27,271,680. These counts would change for SwiGLU, untied output weights, GQA, or other architectural substitutions. Check the instantiated model before recording an experiment.

Train the tokenizer on training documents only and freeze it for all methods. Start with a small equal-budget learning-rate sweep, for example 3e-4, 1e-3, and 3e-3 for AdamW, then lock the recipe before final seeds. These are starting candidates, not validated settings. Keep all tuning on a development partition, with a separate untouched final test set.

The primary 7.35M model has approximately 14.69 MB of BF16 weight payload, excluding code, metadata, tokenizer, optimizer state, and activations. Its small size incidentally resembles the storage scale of Parameter Golf, but it is not a claim of compliance with the official artifact rules. Actual VRAM use depends heavily on microbatch size and execution path.

A minimal PyTorch training implementation inspired by llama2.c's small-model training pipeline is suitable; its C inference engine is not required. Parameter Golf supplies useful budget and evaluation ideas, while Zoology supplies synthetic memory tasks. Avoid assuming that their original configurations and losses transfer unchanged to this proposed model. [llama2.c training guide](https://github.com/karpathy/llama2.c#training).

**What “gradient-free” means in this experiment.** Record the claim precisely for every method.

| Category | Allowed operations | Interpretation |
|---|---|---|
| Local correlation/count learning | Fixed/random starting features, competitive Hebbian/Oja updates, normalized count or prototype readout | No end-to-end gradient training; identify which rules have an optimization interpretation |
| Explicit local error correction | Closed-form delta updates using a local target | No autograd or global backpropagation, but a delta rule can mathematically be a gradient step |
| Gradient-trained representations with local adaptation | Shared pretrained backbone plus Hebbian memory | Gradient-free adaptation only; pretraining is not gradient-free |
| Transformer control | End-to-end backpropagation and AdamW | Conventional from-scratch language-model baseline |

A method is not genuinely independent of global backpropagation merely because the source code avoids calling backward(). Hand-coded global derivatives or the sign of a backpropagated gradient belong to the gradient-based category. Likewise, replacing AdamW with a Hebbian update in an otherwise unchanged transformer does not by itself solve feature learning, output prediction, and deep credit assignment. [Fast-weight/delta connection](https://arxiv.org/abs/2102.11174), [Global-guided Hebbian Learning](https://arxiv.org/abs/2601.21367).

**First comparison set.** Start with inexpensive controls and a paired Hebbian intervention:

1. A smoothed n-gram count model. This establishes whether the neural methods do more than exploit short local statistics. Count its stored tables toward persistent memory.
2. A fixed random context encoder plus a gradient-free count/prototype readout. This measures how far fixed features and supervised association alone get.
3. The same context encoder augmented with competitive Hebbian feature learning, using the same readout procedure. This isolates the contribution of learned local features.
4. An explicitly labeled local delta-rule variant, particularly for memory and readout experiments. This tests error correction without claiming mathematical absence of gradients.
5. The 7.35M transformer trained with AdamW. This is the overall language-model reference, not a claim that every other architecture is identical.

**Priority for the first implementation.** Implement the transformer, n-gram control, and the paired random/Hebbian feature models first. Defer the delta variant until that pair is working. This gives four systems, with only three neural systems requiring a GPU screen, and answers two questions: whether local feature learning adds value, and how far the complete system is from a conventional LM.

For the Hebbian candidate, start with one competitive feature layer using normalized/Oja-style local updates, followed by the same smoothed association readout used by the random-feature control. Use an explicitly causal context encoder with a maximum 512-token receptive field and identical initialization in each paired seed. Specify its order encoding before implementation; a bag of context tokens loses information that a transformer can use. Freeze the feature layer in the control, and update it from observed context only in the Hebbian candidate. Neither system receives pretrained embeddings. This is a proposed research architecture, not a published competitive Hebbian language model. SoftHebb is an inspiration for competitive local feature learning; its reported vision results do not establish the effectiveness of this language adaptation. [SoftHebb paper](https://arxiv.org/abs/2209.11883).

The remaining shortlist has distinct purposes:

| Candidate | Priority | What it would establish |
|---|---|---|
| Local delta readout or memory | Next ablation | Whether local error correction improves on correlation-only updates; label it no-global-backprop, since the delta rule can be a gradient step |
| Small linear-attention LM, then Gated DeltaNet if justified | Second round, trained with backpropagation | Whether a fast associative memory architecture works when representation learning is supplied by conventional optimization |
| Shared trained transformer with Hebbian or delta memory | Separate adaptation track | Whether local writes improve recall or adaptation after conventional pretraining |

Linear-attention fast weights and delta memories are relevant bridges between these tracks. Their published formulations use gradient-trained controllers; an outer-product memory write does not make their entire training procedure gradient-free. A smaller implementation is itself an experiment, so published larger-model results are motivation rather than a prediction of its score. [Fast Weight Programmers](https://arxiv.org/abs/2102.11174), [Gated DeltaNet](https://arxiv.org/abs/2412.06464).

Do not require every candidate to have exactly 7.35M parameters. The paired random/Hebbian models must match feature dimensions and readout capacity, while the cross-architecture comparison should report the actual quality, training time, persistent storage, and inference state of each system. In particular, dense vocabulary association tables can dominate the size of a nominally small Hebbian model.

**First round execution decision.** Once implementation is requested, first validate the data/scorer with the 1.31M debug transformer and measure the 7.35M model's throughput in a two-minute warm pilot. Then screen each neural system with up to three configurations of at most 15 minutes each, using one screening seed and the same development data. This is up to 135 minutes of neural training across three systems, excluding preparation, compilation, evaluation, and the count control. Use the same candidate training stream and record how much each run consumes. Token exposure matching and time matching remain separate comparisons.

Advance the transformer and best eligible local candidate to three seeds with a 60-minute training cap per seed: up to six further GPU-hours, excluding other costs. Include the corresponding random-feature control in final seed runs if claiming a reproducible benefit from Hebbian features; that adds up to three hours. Treat these as compute caps, not estimates of total engineering time or evidence that all runs can finish 100M tokens. If the feature learner loses to its frozen control, investigate that result before expanding to larger models or FineWeb.

The count/prototype readout must define a normalized probability over the full vocabulary so it can be scored by cross-entropy. One simple version uses nonnegative normalized feature activities to mix smoothed per-feature token-frequency distributions. Accumulate a feature/token association only after the target token has been revealed. Select smoothing on development data, never on test outcomes. For the paired fixed/Hebbian experiment, keep feature widths, normalization, and target exposure identical.

For memory-only diagnostics, compare additive outer-product writes, stabilized/normalized Hebbian writes, and error-correcting delta writes. Include exact key lookup as a sanity-check ceiling, not as a learned competitor. Cap fast-state memory and report its actual size. For learned MQAR models, provide identical token sequences and prediction targets; granting a hand-written memory module privileged key/value parsing belongs in a separate diagnostic track.

Two comparisons are deliberately distinct: the paired fixed/Hebbian feature models isolate a learning-rule change, while comparison with a transformer tests practical language-model quality under resources. A win in one does not automatically establish the other. Do not spend the first experiments trying to equalize architectures so strictly that the proposed local method loses the mechanism being tested.

**Evaluation protocol.** Use held-out document splits, one fixed tokenizer, and next-token log loss/perplexity within a corpus. Also compute BPB as total negative log probability in bits divided by the original UTF-8 bytes of the scored text. Handle BOS/EOS and padding consistently, and do not double-count text across overlapping context windows. Use the same scoring positions, context access, and state-reset policy for every method.

Keep learned long-term parameters frozen for the primary held-out evaluation. Normal causal state evolution is allowed: a transformer's KV cache grows, a recurrent hidden state evolves, and a fast-weight architecture can update its designated sequence state. Reset those states at each document boundary. This permits the architecture's forward computation without allowing unrestricted cross-document test-set training.

A separate adaptation experiment can permit changes to selected learned parameters, but must score a chunk before adapting on its revealed tokens. Predict later chunks using the resulting state, reset at the same document boundaries, and count adaptation latency and memory. No future answers may affect an earlier score. OpenAI explicitly highlighted score-first, per-document LoRA TTT as a valid Parameter Golf submission strategy. [OpenAI retrospective](https://openai.com/index/what-parameter-golf-taught-us/).

If a shared gradient-trained backbone is used in this adaptation track, give the same backbone to all memory variants and record its training cost. A locally updated module attached to that backbone is evidence about adaptation, not about training the whole LM without gradients.

Record train/development/test loss, tokens consumed, token throughput, training elapsed time, setup/compilation time, evaluation/adaptation time, peak VRAM, parameter count, persistent artifact bytes, and transient state bytes. For MQAR, report accuracy as a function of memory load, query distance, and overwrite frequency. For final language-model candidates, report mean and dispersion across three fixed seeds. Give each method an explicitly equal tuning budget and log failed runs.

**How long should this take?** Use a two-minute warm training pilot once adequate GPU memory is available to measure throughput for the actual model, batch, and software stack. Then compute the token-budget time directly. For 100M training tokens:

| Measured throughput | Training time, excluding setup/evaluation |
|---|---|
| 10,000 tokens/s | 166.7 minutes |
| 20,000 tokens/s | 83.3 minutes |
| 50,000 tokens/s | 33.3 minutes |
| 100,000 tokens/s | 16.7 minutes |

These are arithmetic examples, not a forecast that the RTX 4090 will achieve a particular row. A Hebbian implementation with serial per-token work may be slower than an optimized transformer despite omitting a backward pass. A useful local budget is 15 minutes for screening and 60 minutes for a main run, with final candidates extended to equal token counts when needed. Five methods at three one-hour seeds consume 15 GPU-hours before tuning, preprocessing, and evaluation; narrow the field with the cheaper screens first.

**A useful initial success criterion.** First require that the learned Hebbian features improve over the matching random-feature control and that the full system beats a reasonable count baseline on held-out text. Next test whether it approaches the transformer at equal data exposure, or provides a worthwhile quality–time–memory tradeoff. Finally confirm any gain on FineWeb and across seeds. Strong MQAR performance with weak TinyStories performance identifies a representation or credit-assignment problem; it does not establish a replacement for language-model training.
