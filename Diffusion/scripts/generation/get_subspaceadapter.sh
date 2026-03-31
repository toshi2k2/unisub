#!/bin/bash

# Compute SubspaceAdapter components for all LoRAs in lora_names.json.
# Source and target LoRAs are both read from lora_names.json (default).
# Use --output_type to switch between 'reconstruction' (default) and 'subspaceadapter'.

python get_subspaceadapter.py \
  --source_lora_config "lora_names.json" \
  --num_eigenvector_components 32 \
  --output_type "reconstruction" \
  --output_dir "./output"
