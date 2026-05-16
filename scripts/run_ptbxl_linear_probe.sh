#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HEARTLANG_REPO_URL="${HEARTLANG_REPO_URL:-https://github.com/PKUDigitalHealth/HeartLang.git}"
HEARTLANG_DIR="${HEARTLANG_DIR:-$PROJECT_ROOT/external/HeartLang}"
PTBXL_DOWNLOAD_DIR="${PTBXL_DOWNLOAD_DIR:-$PROJECT_ROOT/data/ptb-xl}"
PTBXL_ROOT="${PTBXL_ROOT:-$PTBXL_DOWNLOAD_DIR/physionet.org/files/ptb-xl/1.0.3}"

TASK="${TASK:-superdiagnostic}"
TRAINABLE="${TRAINABLE:-linear}"
SPLIT_RATIOS="${SPLIT_RATIOS:-1}"
SAMPLING_METHOD="${SAMPLING_METHOD:-random}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-5e-3}"
SEED="${SEED:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USED_CHANNELS="${USED_CHANNELS:-0 1 2 3 4 5 6 7 8 9 10 11}"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
DOWNLOAD_PTBXL="${DOWNLOAD_PTBXL:-1}"
DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-1}"
RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<'USAGE'
Run HeartLang PTB-XL linear probing on a GPU machine.

Default run:
  bash scripts/run_ptbxl_linear_probe.sh

Common overrides:
  TASK=form SPLIT_RATIOS="0.01 0.1 1" bash scripts/run_ptbxl_linear_probe.sh
  TASK=rhythm EPOCHS=50 RUN_PREPROCESS=0 bash scripts/run_ptbxl_linear_probe.sh
  TRAINABLE=all TASK=superdiagnostic bash scripts/run_ptbxl_linear_probe.sh

Supported TASK values:
  superdiagnostic, subdiagnostic, form, rhythm

Important:
  Run this on the 4090 machine, not on the Mac. PTB-XL data, checkpoints,
  logs, and outputs are intentionally ignored by Git.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
}

nb_classes_for_task() {
  case "$1" in
    superdiagnostic) echo 5 ;;
    subdiagnostic) echo 23 ;;
    form) echo 19 ;;
    rhythm) echo 12 ;;
    *)
      echo "Unsupported TASK='$1'. Use: superdiagnostic, subdiagnostic, form, rhythm." >&2
      exit 1
      ;;
  esac
}

random_free_port() {
  python - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
}

clone_heartlang() {
  require_command git

  if [[ ! -d "$HEARTLANG_DIR/.git" ]]; then
    mkdir -p "$(dirname "$HEARTLANG_DIR")"
    git clone "$HEARTLANG_REPO_URL" "$HEARTLANG_DIR"
  else
    echo "Using existing HeartLang checkout: $HEARTLANG_DIR"
  fi
}

install_deps() {
  if [[ "$INSTALL_DEPS" != "1" ]]; then
    echo "Skipping dependency installation because INSTALL_DEPS=$INSTALL_DEPS"
    return
  fi

  require_command python
  python -m pip install -U pip
  python -m pip install -r "$HEARTLANG_DIR/requirements.txt"
  python -m pip install -U huggingface_hub wfdb scikit-learn pandas numpy
}

download_ptbxl() {
  if [[ "$DOWNLOAD_PTBXL" != "1" ]]; then
    echo "Skipping PTB-XL download because DOWNLOAD_PTBXL=$DOWNLOAD_PTBXL"
    return
  fi

  if [[ -f "$PTBXL_ROOT/ptbxl_database.csv" ]]; then
    echo "PTB-XL already exists: $PTBXL_ROOT"
    return
  fi

  require_command wget
  mkdir -p "$PTBXL_DOWNLOAD_DIR"
  wget -r -N -c -np -P "$PTBXL_DOWNLOAD_DIR" https://physionet.org/files/ptb-xl/1.0.3/
}

download_checkpoints() {
  if [[ "$DOWNLOAD_CHECKPOINTS" != "1" ]]; then
    echo "Skipping checkpoint download because DOWNLOAD_CHECKPOINTS=$DOWNLOAD_CHECKPOINTS"
    ensure_official_checkpoint_layout
    return
  fi

  if [[ -f "$HEARTLANG_DIR/checkpoints/heartlang_base/checkpoint-200.pth" ]]; then
    echo "HeartLang checkpoint already exists."
    return
  fi

  python -m pip install -U huggingface_hub
  huggingface-cli download PKUDigitalHealth/HeartLang --local-dir "$HEARTLANG_DIR"
  ensure_official_checkpoint_layout
}

ensure_official_checkpoint_layout() {
  local hf_checkpoint="$HEARTLANG_DIR/checkpoints/pretrain/MIMIC-IV/checkpoint-200.pth"
  local official_checkpoint_dir="$HEARTLANG_DIR/checkpoints/heartlang_base"
  local official_checkpoint="$official_checkpoint_dir/checkpoint-200.pth"

  if [[ -f "$official_checkpoint" ]]; then
    return
  fi

  if [[ -f "$hf_checkpoint" ]]; then
    mkdir -p "$official_checkpoint_dir"
    ln -sf ../pretrain/MIMIC-IV/checkpoint-200.pth "$official_checkpoint"
    return
  fi
}

patch_ptbxl_path() {
  PTBXL_ROOT_FOR_PATCH="$PTBXL_ROOT" python - <<'PY'
import os
import re
from pathlib import Path

path = Path("datasets/dataset_preprocess/PTBXL/ecg_ptbxl_benchmarking/code/reproduce_results.py")
ptbxl_root = os.environ["PTBXL_ROOT_FOR_PATCH"].rstrip("/") + "/"
text = path.read_text()
text = re.sub(r'datafolder = ".*"', f'datafolder = "{ptbxl_root}"', text, count=1)
path.write_text(text)
PY
}

prepare_ptbxl() {
  if [[ "$RUN_PREPROCESS" != "1" ]]; then
    echo "Skipping preprocessing because RUN_PREPROCESS=$RUN_PREPROCESS"
    return
  fi

  if [[ ! -f "$PTBXL_ROOT/ptbxl_database.csv" ]]; then
    echo "PTB-XL root is missing ptbxl_database.csv: $PTBXL_ROOT" >&2
    echo "Set PTBXL_ROOT=/path/to/ptb-xl/1.0.3 or keep DOWNLOAD_PTBXL=1." >&2
    exit 1
  fi

  cd "$HEARTLANG_DIR"
  patch_ptbxl_path
  python datasets/dataset_preprocess/PTBXL/ecg_ptbxl_benchmarking/code/reproduce_results.py
  python datasets/dataset_preprocess/PTBXL/move_ptbxl_files.py
  python QRSTokenizer.py --dataset_name PTBXL --used_channels $USED_CHANNELS
  copy_ptbxl_labels
}

copy_ptbxl_labels() {
  local source_root="$HEARTLANG_DIR/datasets/ecg_datasets/PTBXL"
  local target_root="$HEARTLANG_DIR/datasets/ecg_datasets/PTBXL_QRS"
  local subdirs=(all diagnostic form rhythm subdiagnostic superdiagnostic)
  local prefixes=(train val test)

  for subdir in "${subdirs[@]}"; do
    mkdir -p "$target_root/$subdir"
    for prefix in "${prefixes[@]}"; do
      local source_file="$source_root/$subdir/${prefix}_labels.npy"
      if [[ -f "$source_file" ]]; then
        cp "$source_file" "$target_root/$subdir/${prefix}_labels.npy"
      fi
    done
  done
}

run_training_and_eval() {
  local nb_classes
  nb_classes="$(nb_classes_for_task "$TASK")"

  local finetune_ckpt="$HEARTLANG_DIR/checkpoints/heartlang_base/checkpoint-200.pth"
  if [[ ! -f "$finetune_ckpt" ]]; then
    echo "Missing checkpoint: $finetune_ckpt" >&2
    echo "Keep DOWNLOAD_CHECKPOINTS=1 or set up the checkpoint before running." >&2
    exit 1
  fi

  local dataset_dir="datasets/ecg_datasets/PTBXL_QRS/$TASK"
  if [[ ! -f "$HEARTLANG_DIR/$dataset_dir/train_data.npy" ]]; then
    echo "Missing tokenized PTB-XL data under $HEARTLANG_DIR/$dataset_dir" >&2
    echo "Keep RUN_PREPROCESS=1 for the first run." >&2
    exit 1
  fi

  cd "$HEARTLANG_DIR"

  for ratio in $SPLIT_RATIOS; do
    local run_name="finetune_${TASK}_base_${TRAINABLE}_${ratio}_${SAMPLING_METHOD}"
    local output_dir="checkpoints/finetune/ptbxl/$run_name"

    if [[ "$RUN_TRAIN" == "1" ]]; then
      local train_port
      train_port="$(random_free_port)"
      torchrun --nnodes=1 --master_port="$train_port" --nproc_per_node=1 run_class_finetuning.py \
        --dataset_dir "$dataset_dir" \
        --output_dir "$output_dir" \
        --log_dir "log/finetune/$run_name" \
        --model HeartLang_finetune_base \
        --finetune "$finetune_ckpt" \
        --trainable "$TRAINABLE" \
        --split_ratio "$ratio" \
        --sampling_method "$SAMPLING_METHOD" \
        --weight_decay 0.05 \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --update_freq 1 \
        --warmup_epochs 10 \
        --epochs "$EPOCHS" \
        --layer_decay 0.9 \
        --save_ckpt_freq "$EPOCHS" \
        --seed "$SEED" \
        --is_binary \
        --nb_classes "$nb_classes" \
        --world_size 1
    fi

    if [[ "$RUN_EVAL" == "1" ]]; then
      local eval_port
      eval_port="$(random_free_port)"
      torchrun --nnodes=1 --master_port="$eval_port" --nproc_per_node=1 run_class_finetuning.py \
        --dataset_dir "$dataset_dir" \
        --output_dir "$output_dir" \
        --log_dir "log/finetune_test/$run_name" \
        --model HeartLang_finetune_base \
        --eval \
        --trainable "$TRAINABLE" \
        --split_ratio "$ratio" \
        --sampling_method "$SAMPLING_METHOD" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        --is_binary \
        --nb_classes "$nb_classes" \
        --world_size 1
    fi
  done
}

main() {
  require_command python
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found. Continue only if this machine has CUDA configured."
  fi

  clone_heartlang
  install_deps
  download_ptbxl
  download_checkpoints
  prepare_ptbxl
  run_training_and_eval
}

main "$@"
