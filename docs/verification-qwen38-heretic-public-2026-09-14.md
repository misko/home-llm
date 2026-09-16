# Public Qwen3.8 Heretic artifact verification

Verified on 2026-09-14 with the Lab data root at `/mnt/md2/llm-lab`.

## RVN Q4_K_M

- Repository: `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF`
- Resolved revision: `20b94f0613b632b4848bbe3b1e05d9ee0c2b1608`
- Selected weight: `RVN-Q4_K_M-multilingual.gguf`
- Weight SHA-256: `9ed40ccc8b8432f38b9a85d0ca67928167f5719e8c10bd299e56d34facaf6e61`
- Logical artifact size: 16,547,443,996 bytes
- Lab manifest: `be21e870667217997f5e227df539c7d5de70143ef7104ddf461e423a3d102ce3`
- Registry alias: `heretic-rvn-public`
- Deployment alias: `local-heretic-rvn`

`llmctl artifact verify` passed for all three selected files. The pinned
llama.cpp runtime loaded the model completely on the RTX 4090, exposed a healthy
OpenAI-compatible endpoint, and reported 26,895,998,464 parameters, an 8,192
token serving context, a 262,144 token training context, and Q4_K_M weights. A
deterministic chat request asking for `RVN_OK` returned exactly `RVN_OK`.

The prior `local-fable` deployment was restored after the smoke test and passed
its health check.

## ARA BF16

- Repository: `trohrbaugh/Qwen3.8-27B-heretic-ara`
- Resolved revision: `a67ae100d933c0d17af3232bda35825979fc63ce`
- Format: seven BF16 safetensor files plus configuration, tokenizer, template,
  preprocessors, and model card
- Logical artifact size: 55,583,189,608 bytes
- Lab manifest: `7c6439f57e1a972aca5cf235202b1846983da91e6c366acf1232776f6726bba9`
- Registry alias: `heretic-ara-bf16`

`llmctl artifact verify` passed for all sixteen selected files. Each of the
seven weight-file SHA-256 values matched the pinned Hugging Face LFS digest. A
separate structural validation opened every safetensor file, confirmed that all
1,199 entries in `model.safetensors.index.json` exist in the declared shard with
no extra tensors, loaded the local model configuration and tokenizer, and
successfully rendered a non-thinking chat prompt. The configuration reports
`qwen3_5`, BF16, and a vocabulary size of 248,320.

The full checkpoint is retained for research, conversion, and CPU/offloaded
validation. Its 51.77 GiB installed size prevents direct full-GPU serving on the
24 GB RTX 4090; the verified RVN GGUF is the ready-to-serve artifact.
