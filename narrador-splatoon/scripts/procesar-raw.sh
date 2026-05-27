#!/usr/bin/env bash

set -euo pipefail

RAW_DIR="./dataset/raw"
PENDING_DIR="./dataset/pending_tag"

mkdir -p "$PENDING_DIR"

echo "Scanning raw dataset folder: $RAW_DIR"
echo "Output folder: $PENDING_DIR"

# Copy JPEG images
find "$RAW_DIR" -type f \( \
    -iname "*.jpg" -o \
    -iname "*.jpeg" \
\) -print0 | while IFS= read -r -d '' image_file; do
    filename="$(basename "$image_file")"
    destination="$PENDING_DIR/$filename"

    cp "$image_file" "$destination"
    echo "Copied image: $image_file -> $destination"
done

# Extract one frame every 0.5 seconds from MP4 videos
find "$RAW_DIR" -type f -iname "*.mov" -print0 | while IFS= read -r -d '' video_file; do
    video_name="$(basename "$video_file")"
    video_stem="${video_name%.*}"

    output_pattern="$PENDING_DIR/${video_stem}_frame_%06d.jpg"

    ffmpeg \
        -nostdin \
        -hide_banner \
        -loglevel error \
        -i "$video_file" \
        -vf "fps=2,scale=1280:720" \
        -q:v 2 \
        "$output_pattern"

    echo "Extracted frames from: $video_file"
done

echo "Done."