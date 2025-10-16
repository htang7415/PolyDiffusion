#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$ROOT/quickstart_artifacts"
DATADIR="$WORKDIR/data"
export DATADIR
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p "$DATADIR"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATADIR"])
stage_a = root / "stage_a.jsonl"
stage_b = root / "stage_b.jsonl"
stage_c = root / "stage_c.csv"

stage_a_records = [
    {"smiles": "CCO", "synth_score": 3.2},
    {"smiles": "CCC", "synth_score": 3.8},
    {"smiles": "CCN", "synth_score": 2.7},
]
stage_b_records = [
    {"ap_smiles": "[*:1]CCO[*:2]", "synth_score": 3.6},
    {"ap_smiles": "[*:1]CCN[*:2]", "synth_score": 3.1},
]
stage_c_records = [
    {
        "ap_smiles": "[*:1]CCO[*:2]",
        "synth_score": 3.6,
        "Tg": 210,
        "Tm": 320,
        "Td": 520,
        "Eg": 3.4,
        "chi": 0.45,
    },
    {
        "ap_smiles": "[*:1]CCN[*:2]",
        "synth_score": 3.1,
        "Tg": 205,
        "Tm": 315,
        "Td": 510,
        "Eg": 3.5,
        "chi": 0.47,
    },
]

with stage_a.open("w", encoding="utf-8") as handle:
    for rec in stage_a_records:
        handle.write(json.dumps(rec) + "\n")

with stage_b.open("w", encoding="utf-8") as handle:
    for rec in stage_b_records:
        handle.write(json.dumps(rec) + "\n")

with stage_c.open("w", encoding="utf-8") as handle:
    handle.write("ap_smiles,synth_score,Tg,Tm,Td,Eg,chi\n")
    for rec in stage_c_records:
        handle.write(
            f"{rec['ap_smiles']},{rec['synth_score']},{rec['Tg']},{rec['Tm']},{rec['Td']},{rec['Eg']},{rec['chi']}\n"
        )
PY

VOCAB_PATH="$WORKDIR/vocab.txt"
python -m PolyDiffusion.scripts.build_vocab "$DATADIR/stage_b.jsonl" "$VOCAB_PATH" --field ap_smiles --limit 100

create_config() {
  local path="$1"
  local data_path="$2"
  local steps="$3"
  local ckpt_name="$(basename "$path" .yaml).pt"
  cat >"$path" <<EOF
vocab_path: $VOCAB_PATH
model_config: $ROOT/configs/model_base.yaml
data:
  path: $data_path
training:
  batch_size: 2
  lr: 0.001
  steps: $steps
  log_interval: 1
loss:
  lambda_syn: 0.5
  lambda_prop: 1.0
  lambda_gram: 0.1
checkpoint_path: $WORKDIR/$ckpt_name
EOF
}

create_config "$WORKDIR/stage_a.yaml" "$DATADIR/stage_a.jsonl" 3
create_config "$WORKDIR/stage_b.yaml" "$DATADIR/stage_b.jsonl" 3
create_config "$WORKDIR/stage_c.yaml" "$DATADIR/stage_c.csv" 3

PYTHONPATH="$ROOT" python -m PolyDiffusion.train.train_stage_a --config "$WORKDIR/stage_a.yaml"
PYTHONPATH="$ROOT" python -m PolyDiffusion.train.train_stage_b --config "$WORKDIR/stage_b.yaml"
PYTHONPATH="$ROOT" python -m PolyDiffusion.train.train_stage_c --config "$WORKDIR/stage_c.yaml"

PYTHONPATH="$ROOT" python -m PolyDiffusion.scripts.sample_cli \
  --ckpt "$WORKDIR/stage_c.pt" \
  --vocab "$VOCAB_PATH" \
  --config "$ROOT/configs/model_base.yaml" \
  --num 5 \
  --steps 3 \
  --s_target 3.5
