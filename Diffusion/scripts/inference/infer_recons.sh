#!/bin/bash

# Generate images for all LoRAs in lora_names.json using Reconstructed LoRA weights.

python sdxl_inference.py \
  --lora_names_config "lora_names.json" \
  --adapter_dir "./output" \
  --output_type "reconstruction"
