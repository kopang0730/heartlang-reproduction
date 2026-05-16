#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HEARTLANG_DIR="${HEARTLANG_DIR:-$PROJECT_ROOT/external/HeartLang}"
EXPORT_ROOT="${EXPORT_ROOT:-$PROJECT_ROOT/exports}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_NAME="${BUNDLE_NAME:-heartlang_results_$STAMP}"
BUNDLE_DIR="$EXPORT_ROOT/$BUNDLE_NAME"
ARCHIVE_PATH="$EXPORT_ROOT/$BUNDLE_NAME.tar.gz"

copy_tree_if_exists() {
  local source_rel="$1"
  local source="$HEARTLANG_DIR/$source_rel"

  if [[ -e "$source" ]]; then
    mkdir -p "$BUNDLE_DIR"
    tar \
      --exclude='*.pth' \
      --exclude='*.pt' \
      --exclude='*.ckpt' \
      --exclude='*.safetensors' \
      -C "$HEARTLANG_DIR" \
      -cf - "$source_rel" | tar -C "$BUNDLE_DIR" -xf -
  fi
}

write_manifest() {
  {
    echo "HeartLang result bundle"
    echo "Created at: $(date -Is)"
    echo "Project root: $PROJECT_ROOT"
    echo "HeartLang dir: $HEARTLANG_DIR"
    echo
    echo "Included paths:"
    find "$BUNDLE_DIR" -maxdepth 4 -type f | sed "s#^$BUNDLE_DIR/##" | sort
  } > "$BUNDLE_DIR/MANIFEST.txt"
}

main() {
  if [[ ! -d "$HEARTLANG_DIR" ]]; then
    echo "Missing HeartLang directory: $HEARTLANG_DIR" >&2
    echo "Run scripts/run_ptbxl_linear_probe.sh first, or set HEARTLANG_DIR." >&2
    exit 1
  fi

  mkdir -p "$BUNDLE_DIR"

  # Evaluation summaries, prediction probabilities, and target labels.
  copy_tree_if_exists "results"
  copy_tree_if_exists "results/pred"

  # Training logs and lightweight checkpoint metadata/log.txt files.
  copy_tree_if_exists "log"
  copy_tree_if_exists "checkpoints/finetune/ptbxl"

  python "$PROJECT_ROOT/scripts/visualize_ptbxl_results.py" \
    --heartlang-dir "$HEARTLANG_DIR" \
    --output-dir "$BUNDLE_DIR/figures"

  write_manifest

  mkdir -p "$EXPORT_ROOT"
  tar -czf "$ARCHIVE_PATH" -C "$EXPORT_ROOT" "$BUNDLE_NAME"

  echo
  echo "Result archive created:"
  echo "$ARCHIVE_PATH"
  echo
  echo "On your Mac, download with scp, for example:"
  echo "scp -P <AutoDL_SSH_PORT> root@<AutoDL_HOST>:$ARCHIVE_PATH ~/Downloads/"
}

main "$@"
