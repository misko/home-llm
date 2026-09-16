"""CPU-only calibration gate for the independent development entailment model."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_lab.comparison.contracts import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judge", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.judge, local_files_only=True).to("cpu").eval()
    entailment_id = model.config.label2id["entailment"]
    rows = json.loads(args.cases.read_text())
    results = []
    for row in rows:
        inputs = tokenizer(row["premise"], row["hypothesis"], return_tensors="pt")
        with torch.inference_mode():
            probability = model(**inputs).logits.softmax(-1)[0, entailment_id].item()
        results.append({**row, "probability":probability, "correct":(probability >= .7) == row["entails"]})
    recall = sum(r["correct"] for r in results if r["entails"]) / sum(r["entails"] for r in results)
    specificity = sum(r["correct"] for r in results if not r["entails"]) / sum(not r["entails"] for r in results)
    passed = recall >= .9 and specificity >= .9
    result = {"model":"cross-encoder/nli-deberta-v3-small", "revision":"fa2804872c3b4bd748f38c0185cc85775361e735", "threshold":.7,
              "entailment_id":entailment_id, "calibration_sha256":file_hash(args.cases), "calibration_passed":passed,
              "recall":recall, "specificity":specificity, "results":results,
              "files":{p.name:file_hash(p) for p in args.judge.iterdir() if p.is_file()}}
    atomic_json(args.output, result)
    print(json.dumps({k:result[k] for k in ["calibration_passed","recall","specificity"]}))
    if not passed:
        raise SystemExit("Semantic scorer failed calibration; do not use it for selection.")


if __name__ == "__main__":
    main()
