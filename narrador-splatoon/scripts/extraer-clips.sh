#!/usr/bin/env bash

set -euo pipefail

SEARCH_DIR="${1:-.}"
DATASET_DIR="./dataset/raw"
PATTERN="4CE9651EE88A979D41F24CE8D6EA1C23"

mkdir -p "$DATASET_DIR"

find "$SEARCH_DIR" -type f -name "*${PATTERN}*" -print0 | while IFS= read -r -d '' file; do
    filename="$(basename "$file")"

    # Skip files already inside the dataset folder
    if [[ "$file" == "$DATASET_DIR/"* || "$file" == ./"$DATASET_DIR/"* ]]; then
        continue
    fi

    cp "$file" "$DATASET_DIR/$filename"
    echo "Copied: $file -> $DATASET_DIR/$filename"
done