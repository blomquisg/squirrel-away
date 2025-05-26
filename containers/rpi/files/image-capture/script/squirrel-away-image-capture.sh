#!/bin/bash

# some strict error handling
set -euo pipefail

DESTINATION_DIR="/var/lib/squirrel-away/inference/input"
mkdir -p $DESTINATION_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="$TIMESTAMP.jpg"

# Capture a single image and write it out
libcamera-still --output "$DESTINATION_DIR/$FILENAME" --timeout 1 --nopreview
