# Diffusers Integration

This `diffusers/` tree contains the local diffusion adapter integration used by the UniSub diffusion experiments.

The main experiment workflow itself lives in [../Diffusion/README.md](../Diffusion/README.md). This directory holds the supporting loader and state-dict plumbing needed to make those adapter experiments work.

Useful entry points:

- [utils/state_dict_utils.py](./utils/state_dict_utils.py): mappings for LoRA and SubspaceAdapter state dict conversions
- [loaders/peft.py](./loaders/peft.py): PEFT adapter loading and activation mixins
- [loaders/lora_base.py](./loaders/lora_base.py): LoRA loading and adapter management helpers

If you are looking for the more focused PEFT project rather than this local integration layer, use [EigenLoRA](https://github.com/toshi2k2/EigenLoRA/).
