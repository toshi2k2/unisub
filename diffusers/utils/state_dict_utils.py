# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
State dict utilities: utility methods for converting state dicts easily
"""

import enum

from .logging import get_logger


logger = get_logger(__name__)


class StateDictType(enum.Enum):
    """
    The mode to use when converting state dicts.
    """

    DIFFUSERS_OLD = "diffusers_old"
    KOHYA_SS = "kohya_ss"
    PEFT = "peft"
    PEFT_SUBSPACEADAPTER = "peft_subspaceadapter"
    DIFFUSERS = "diffusers"
    DIFFUSERS_SUBSPACEADAPTER = "diffusers_subspaceadapter"


# We need to define a proper mapping for Unet since it uses different output keys than text encoder
# e.g. to_q_lora -> q_proj / to_q
UNET_TO_DIFFUSERS = {
    ".to_out_lora.up": ".to_out.0.lora_B",
    ".to_out_lora.down": ".to_out.0.lora_A",
    ".to_q_lora.down": ".to_q.lora_A",
    ".to_q_lora.up": ".to_q.lora_B",
    ".to_k_lora.down": ".to_k.lora_A",
    ".to_k_lora.up": ".to_k.lora_B",
    ".to_v_lora.down": ".to_v.lora_A",
    ".to_v_lora.up": ".to_v.lora_B",
    ".lora.up": ".lora_B",
    ".lora.down": ".lora_A",
    ".to_out.lora_magnitude_vector": ".to_out.0.lora_magnitude_vector",
}

UNET_TO_DIFFUSERS_subspaceadapter = {
    ".to_out_subspaceadapter.up": ".to_out.0.subspaceadapter_B",
    ".to_out_subspaceadapter.down": ".to_out.0.subspaceadapter_A",
    ".to_q_subspaceadapter.down": ".to_q.subspaceadapter_A",
    ".to_q_subspaceadapter.up": ".to_q.subspaceadapter_B",
    ".to_k_subspaceadapter.down": ".to_k.subspaceadapter_A",
    ".to_k_subspaceadapter.up": ".to_k.subspaceadapter_B",
    ".to_v_subspaceadapter.down": ".to_v.subspaceadapter_A",
    ".to_v_subspaceadapter.up": ".to_v.subspaceadapter_B",
    ".subspaceadapter.up": ".subspaceadapter_B",
    ".subspaceadapter.down": ".subspaceadapter_A",
}


DIFFUSERS_TO_PEFT = {
    ".q_proj.lora_linear_layer.up": ".q_proj.lora_B",
    ".q_proj.lora_linear_layer.down": ".q_proj.lora_A",
    ".k_proj.lora_linear_layer.up": ".k_proj.lora_B",
    ".k_proj.lora_linear_layer.down": ".k_proj.lora_A",
    ".v_proj.lora_linear_layer.up": ".v_proj.lora_B",
    ".v_proj.lora_linear_layer.down": ".v_proj.lora_A",
    ".out_proj.lora_linear_layer.up": ".out_proj.lora_B",
    ".out_proj.lora_linear_layer.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
    "text_projection.lora.down.weight": "text_projection.lora_A.weight",
    "text_projection.lora.up.weight": "text_projection.lora_B.weight",
}


DIFFUSERS_OLD_TO_PEFT = {
    ".to_q_lora.up": ".q_proj.lora_B",
    ".to_q_lora.down": ".q_proj.lora_A",
    ".to_k_lora.up": ".k_proj.lora_B",
    ".to_k_lora.down": ".k_proj.lora_A",
    ".to_v_lora.up": ".v_proj.lora_B",
    ".to_v_lora.down": ".v_proj.lora_A",
    ".to_out_lora.up": ".out_proj.lora_B",
    ".to_out_lora.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

DIFFUSERS_OLD_TO_PEFT_subspaceadapter = {
    ".to_q_subspaceadapter.up.components": ".q_proj.subspaceadapter_B.components",
    ".to_q_subspaceadapter.up.loadings": ".q_proj.subspaceadapter_B.loadings",
    ".to_q_subspaceadapter.down.components": ".q_proj.subspaceadapter_A.components",
    ".to_q_subspaceadapter.down.loadings": ".q_proj.subspaceadapter_A.loadings",
    ".to_k_subspaceadapter.up.components": ".k_proj.subspaceadapter_B.components",
    ".to_k_subspaceadapter.up.loadings": ".k_proj.subspaceadapter_B.loadings",
    ".to_k_subspaceadapter.down.components": ".k_proj.subspaceadapter_A.components",
    ".to_k_subspaceadapter.down.loadings": ".k_proj.subspaceadapter_A.loadings",
    ".to_v_subspaceadapter.up.components": ".v_proj.subspaceadapter_B.components",
    ".to_v_subspaceadapter.up.loadings": ".v_proj.subspaceadapter_B.loadings",
    ".to_v_subspaceadapter.down.components": ".v_proj.subspaceadapter_A.components",
    ".to_v_subspaceadapter.down.loadings": ".v_proj.subspaceadapter_A.loadings",
    ".to_out_subspaceadapter.up.components": ".out_proj.subspaceadapter_B.components",
    ".to_out_subspaceadapter.up.loadings": ".out_proj.subspaceadapter_B.loadings",
    ".to_out_subspaceadapter.down.components": ".out_proj.subspaceadapter_A.components",
    ".to_out_subspaceadapter.down.loadings": ".out_proj.subspaceadapter_A.loadings",
    ".subspaceadapter_linear_layer.up.components": ".subspaceadapter_B.components",
    ".subspaceadapter_linear_layer.up.loadings": ".subspaceadapter_B.loadings",
    ".subspaceadapter_linear_layer.down.components": ".subspaceadapter_A.components",
    ".subspaceadapter_linear_layer.down.loadings": ".subspaceadapter_A.loadings",
}

PEFT_TO_DIFFUSERS = {
    ".q_proj.lora_B": ".q_proj.lora_linear_layer.up",
    ".q_proj.lora_A": ".q_proj.lora_linear_layer.down",
    ".k_proj.lora_B": ".k_proj.lora_linear_layer.up",
    ".k_proj.lora_A": ".k_proj.lora_linear_layer.down",
    ".v_proj.lora_B": ".v_proj.lora_linear_layer.up",
    ".v_proj.lora_A": ".v_proj.lora_linear_layer.down",
    ".out_proj.lora_B": ".out_proj.lora_linear_layer.up",
    ".out_proj.lora_A": ".out_proj.lora_linear_layer.down",
    "to_k.lora_A": "to_k.lora.down",
    "to_k.lora_B": "to_k.lora.up",
    "to_q.lora_A": "to_q.lora.down",
    "to_q.lora_B": "to_q.lora.up",
    "to_v.lora_A": "to_v.lora.down",
    "to_v.lora_B": "to_v.lora.up",
    "to_out.0.lora_A": "to_out.0.lora.down",
    "to_out.0.lora_B": "to_out.0.lora.up",
}

PEFT_TO_DIFFUSERS_subspaceadapter = {
    ".q_proj.subspaceadapter_B.loadings": ".q_proj.subspaceadapter_linear_layer.up.loadings",
    ".q_proj.subspaceadapter_B.components": ".q_proj.subspaceadapter_linear_layer.up.components",
    ".q_proj.subspaceadapter_A.loadings": ".q_proj.subspaceadapter_linear_layer.down.loadings",
    ".q_proj.subspaceadapter_A.components": ".q_proj.subspaceadapter_linear_layer.down.components",
    ".k_proj.subspaceadapter_B.loadings": ".k_proj.subspaceadapter_linear_layer.up.loadings",
    ".k_proj.subspaceadapter_B.components": ".k_proj.subspaceadapter_linear_layer.up.components",
    ".k_proj.subspaceadapter_A.loadings": ".k_proj.subspaceadapter_linear_layer.down.loadings",
    ".k_proj.subspaceadapter_A.components": ".k_proj.subspaceadapter_linear_layer.down.components",
    ".v_proj.subspaceadapter_B.loadings": ".v_proj.subspaceadapter_linear_layer.up.loadings",
    ".v_proj.subspaceadapter_B.components": ".v_proj.subspaceadapter_linear_layer.up.components",
    ".v_proj.subspaceadapter_A.loadings": ".v_proj.subspaceadapter_linear_layer.down.loadings",
    ".v_proj.subspaceadapter_A.components": ".v_proj.subspaceadapter_linear_layer.down.components",
    ".out_proj.subspaceadapter_B.loadings": ".out_proj.subspaceadapter_linear_layer.up.loadings",
    ".out_proj.subspaceadapter_B.components": ".out_proj.subspaceadapter_linear_layer.up.components",
    ".out_proj.subspaceadapter_A.loadings": ".out_proj.subspaceadapter_linear_layer.down.loadings",
    ".out_proj.subspaceadapter_A.components": ".out_proj.subspaceadapter_linear_layer.down.components",
    "to_k.subspaceadapter_A.loadings": "to_k.subspaceadapter.down.loadings",
    "to_k.subspaceadapter_A.components": "to_k.subspaceadapter.down.components",
    "to_k.subspaceadapter_B.loadings": "to_k.subspaceadapter.up.loadings",
    "to_k.subspaceadapter_B.components": "to_k.subspaceadapter.up.components",
    "to_q.subspaceadapter_A.loadings": "to_q.subspaceadapter.down.loadings",
    "to_q.subspaceadapter_A.components": "to_q.subspaceadapter.down.components",
    "to_q.subspaceadapter_B.loadings": "to_q.subspaceadapter.up.loadings",
    "to_q.subspaceadapter_B.components": "to_q.subspaceadapter.up.components",
    "to_v.subspaceadapter_A.loadings": "to_v.subspaceadapter.down.loadings",
    "to_v.subspaceadapter_A.components": "to_v.subspaceadapter.down.components",
    "to_v.subspaceadapter_B.loadings": "to_v.subspaceadapter.up.loadings",
    "to_v.subspaceadapter_B.components": "to_v.subspaceadapter.up.components",
    "to_out.0.subspaceadapter_A.loadings": "to_out.0.subspaceadapter.down.loadings",
    "to_out.0.subspaceadapter_A.components": "to_out.0.subspaceadapter.down.components",
    "to_out.0.subspaceadapter_B.loadings": "to_out.0.subspaceadapter.up.loadings",
    "to_out.0.subspaceadapter_B.components": "to_out.0.subspaceadapter.up.components",
}

DIFFUSERS_OLD_TO_DIFFUSERS = {
    ".to_q_lora.up": ".q_proj.lora_linear_layer.up",
    ".to_q_lora.down": ".q_proj.lora_linear_layer.down",
    ".to_k_lora.up": ".k_proj.lora_linear_layer.up",
    ".to_k_lora.down": ".k_proj.lora_linear_layer.down",
    ".to_v_lora.up": ".v_proj.lora_linear_layer.up",
    ".to_v_lora.down": ".v_proj.lora_linear_layer.down",
    ".to_out_lora.up": ".out_proj.lora_linear_layer.up",
    ".to_out_lora.down": ".out_proj.lora_linear_layer.down",
    ".to_k.lora_magnitude_vector": ".k_proj.lora_magnitude_vector",
    ".to_v.lora_magnitude_vector": ".v_proj.lora_magnitude_vector",
    ".to_q.lora_magnitude_vector": ".q_proj.lora_magnitude_vector",
    ".to_out.lora_magnitude_vector": ".out_proj.lora_magnitude_vector",
}


DIFFUSERS_OLD_TO_DIFFUSERS_subspaceadapter = {
    ".to_q_subspaceadapter.up.components": ".q_proj.subspaceadapter_linear_layer.up.components",
    ".to_q_subspaceadapter.down.components": ".q_proj.subspaceadapter_linear_layer.down.components",
    ".to_k_subspaceadapter.up.components": ".k_proj.subspaceadapter_linear_layer.up.components",
    ".to_k_subspaceadapter.down.components": ".k_proj.subspaceadapter_linear_layer.down.components",
    ".to_v_subspaceadapter.up.components": ".v_proj.subspaceadapter_linear_layer.up.components",
    ".to_v_subspaceadapter.down.components": ".v_proj.subspaceadapter_linear_layer.down.components",
    ".to_out_subspaceadapter.up.components": ".out_proj.subspaceadapter_linear_layer.up.components",
    ".to_out_subspaceadapter.down.components": ".out_proj.subspaceadapter_linear_layer.down.components",
    ".to_q_subspaceadapter.up.loadings": ".q_proj.subspaceadapter_linear_layer.up.loadings",
    ".to_q_subspaceadapter.down.loadings": ".q_proj.subspaceadapter_linear_layer.down.loadings",
    ".to_k_subspaceadapter.up.loadings": ".k_proj.subspaceadapter_linear_layer.up.loadings",
    ".to_k_subspaceadapter.down.loadings": ".k_proj.subspaceadapter_linear_layer.down.loadings",
    ".to_v_subspaceadapter.up.loadings": ".v_proj.subspaceadapter_linear_layer.up.loadings",
    ".to_v_subspaceadapter.down.loadings": ".v_proj.subspaceadapter_linear_layer.down.loadings",
    ".to_out_subspaceadapter.up.loadings": ".out_proj.subspaceadapter_linear_layer.up.loadings",
    ".to_out_subspaceadapter.down.loadings": ".out_proj.subspaceadapter_linear_layer.down.loadings",
}

PEFT_TO_KOHYA_SS = {
    "lora_A": "lora_down",
    "lora_B": "lora_up",
    # This is not a comprehensive dict as kohya format requires replacing `.` with `_` in keys,
    # adding prefixes and adding alpha values
    # Check `convert_state_dict_to_kohya` for more
}

PEFT_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_PEFT,
    StateDictType.DIFFUSERS: DIFFUSERS_TO_PEFT,
    StateDictType.DIFFUSERS_SUBSPACEADAPTER: DIFFUSERS_OLD_TO_PEFT_subspaceadapter,
}

DIFFUSERS_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_DIFFUSERS,
    StateDictType.PEFT: PEFT_TO_DIFFUSERS,
    StateDictType.DIFFUSERS_SUBSPACEADAPTER: DIFFUSERS_OLD_TO_DIFFUSERS_subspaceadapter,
}

KOHYA_STATE_DICT_MAPPINGS = {StateDictType.PEFT: PEFT_TO_KOHYA_SS}

KEYS_TO_ALWAYS_REPLACE = {
    ".processor.": ".",
}


def convert_state_dict(state_dict, mapping):
    r"""
    Simply iterates over the state dict and replaces the patterns in `mapping` with the corresponding values.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        mapping (`dict[str, str]`):
            The mapping to use for conversion, the mapping should be a dictionary with the following structure:
                - key: the pattern to replace
                - value: the pattern to replace with

    Returns:
        converted_state_dict (`dict`)
            The converted state dict.
    """
    converted_state_dict = {}
    for k, v in state_dict.items():
        # First, filter out the keys that we always want to replace
        for pattern in KEYS_TO_ALWAYS_REPLACE.keys():
            if pattern in k:
                new_pattern = KEYS_TO_ALWAYS_REPLACE[pattern]
                k = k.replace(pattern, new_pattern)

        for pattern in mapping.keys():
            if pattern in k:
                new_pattern = mapping[pattern]
                k = k.replace(pattern, new_pattern)
                break
        converted_state_dict[k] = v
    return converted_state_dict


def convert_state_dict_to_peft(state_dict, original_type=None, **kwargs):
    r"""
    Converts a state dict to the PEFT format The state dict can be from previous diffusers format (`OLD_DIFFUSERS`), or
    new diffusers format (`DIFFUSERS`). The method only supports the conversion from diffusers old/new to PEFT for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
    """
    if original_type is None:
        # Old diffusers to PEFT
        if any("to_out_lora" in k for k in state_dict.keys()):
            print("A")
            original_type = StateDictType.DIFFUSERS_OLD
        elif any("subspaceadapter_linear_layer" in k for k in state_dict.keys()):
            print("B")
            original_type = StateDictType.DIFFUSERS_SUBSPACEADAPTER
        elif any("lora_linear_layer" in k for k in state_dict.keys()):
            print("C")
            original_type = StateDictType.DIFFUSERS
        else:
            raise ValueError("Could not automatically infer state dict type")

    if original_type not in PEFT_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f"Original type {original_type} is not supported")

    mapping = PEFT_STATE_DICT_MAPPINGS[original_type]
    return convert_state_dict(state_dict, mapping)


def convert_state_dict_to_diffusers(state_dict, original_type=None, **kwargs):
    r"""
    Converts a state dict to new diffusers format. The state dict can be from previous diffusers format
    (`OLD_DIFFUSERS`), or PEFT format (`PEFT`) or new diffusers format (`DIFFUSERS`). In the last case the method will
    return the state dict as is.

    The method only supports the conversion from diffusers old, PEFT to diffusers new for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
        kwargs (`dict`, *args*):
            Additional arguments to pass to the method.

            - **adapter_name**: For example, in case of PEFT, some keys will be pre-pended
                with the adapter name, therefore needs a special handling. By default PEFT also takes care of that in
                `get_peft_model_state_dict` method:
                https://github.com/huggingface/peft/blob/ba0477f2985b1ba311b83459d29895c809404e99/src/peft/utils/save_and_load.py#L92
                but we add it here in case we don't want to rely on that method.
    """
    peft_adapter_name = kwargs.pop("adapter_name", None)
    if peft_adapter_name is not None:
        peft_adapter_name = "." + peft_adapter_name
    else:
        peft_adapter_name = ""
    print(peft_adapter_name)
    if original_type is None:
        # Old diffusers to PEFT
        if any("to_out_lora" in k for k in state_dict.keys()):
            print("1")
            original_type = StateDictType.DIFFUSERS_OLD
        elif any(f".lora_A{peft_adapter_name}.weight" in k for k in state_dict.keys()):
            print("2")
            original_type = StateDictType.PEFT
        elif any(f".subspaceadapter{peft_adapter_name}" in k for k in state_dict.keys()):
            print("3")
            original_type = StateDictType.DIFFUSERS_SUBSPACEADAPTER
        elif any("lora_linear_layer" in k for k in state_dict.keys()):
            # nothing to do
            print("4")
            return state_dict
        else:
            raise ValueError("Could not automatically infer state dict type")

    if original_type not in DIFFUSERS_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f"Original type {original_type} is not supported")

    mapping = DIFFUSERS_STATE_DICT_MAPPINGS[original_type]
    return convert_state_dict(state_dict, mapping)


def convert_unet_state_dict_to_peft(state_dict, use_subspaceadapter):
    r"""
    Converts a state dict from UNet format to diffusers format - i.e. by removing some keys
    """
    if use_subspaceadapter:
        mapping = UNET_TO_DIFFUSERS_subspaceadapter
    else:
        mapping = UNET_TO_DIFFUSERS
    return convert_state_dict(state_dict, mapping)


def convert_all_state_dict_to_peft(state_dict):
    r"""
    Attempts to first `convert_state_dict_to_peft`, and if it doesn't detect `lora_linear_layer` for a valid
    `DIFFUSERS` LoRA for example, attempts to exclusively convert the Unet `convert_unet_state_dict_to_peft`
    """
    try:
        peft_dict = convert_state_dict_to_peft(state_dict)
    except Exception as e:
        if str(e) == "Could not automatically infer state dict type":
            peft_dict = convert_unet_state_dict_to_peft(state_dict)
        else:
            raise

    if not any("lora_A" in key or "lora_B" in key for key in peft_dict.keys()):
        raise ValueError("Your LoRA was not converted to PEFT")

    return peft_dict


def convert_state_dict_to_kohya(state_dict, original_type=None, **kwargs):
    r"""
    Converts a `PEFT` state dict to `Kohya` format that can be used in AUTOMATIC1111, ComfyUI, SD.Next, InvokeAI, etc.
    The method only supports the conversion from PEFT to Kohya for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
        kwargs (`dict`, *args*):
            Additional arguments to pass to the method.

            - **adapter_name**: For example, in case of PEFT, some keys will be pre-pended
                with the adapter name, therefore needs a special handling. By default PEFT also takes care of that in
                `get_peft_model_state_dict` method:
                https://github.com/huggingface/peft/blob/ba0477f2985b1ba311b83459d29895c809404e99/src/peft/utils/save_and_load.py#L92
                but we add it here in case we don't want to rely on that method.
    """
    try:
        import torch
    except ImportError:
        logger.error(
            "Converting PEFT state dicts to Kohya requires torch to be installed."
        )
        raise

    peft_adapter_name = kwargs.pop("adapter_name", None)
    if peft_adapter_name is not None:
        peft_adapter_name = "." + peft_adapter_name
    else:
        peft_adapter_name = ""

    if original_type is None:
        if any(f".lora_A{peft_adapter_name}.weight" in k for k in state_dict.keys()):
            original_type = StateDictType.PEFT

    if original_type not in KOHYA_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f"Original type {original_type} is not supported")

    # Use the convert_state_dict function with the appropriate mapping
    kohya_ss_partial_state_dict = convert_state_dict(
        state_dict, KOHYA_STATE_DICT_MAPPINGS[StateDictType.PEFT]
    )
    kohya_ss_state_dict = {}

    # Additional logic for replacing header, alpha parameters `.` with `_` in all keys
    for kohya_key, weight in kohya_ss_partial_state_dict.items():
        if "text_encoder_2." in kohya_key:
            kohya_key = kohya_key.replace("text_encoder_2.", "lora_te2.")
        elif "text_encoder." in kohya_key:
            kohya_key = kohya_key.replace("text_encoder.", "lora_te1.")
        elif "unet" in kohya_key:
            kohya_key = kohya_key.replace("unet", "lora_unet")
        elif "lora_magnitude_vector" in kohya_key:
            kohya_key = kohya_key.replace("lora_magnitude_vector", "dora_scale")

        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_key = kohya_key.replace(peft_adapter_name, "")  # Kohya doesn't take names
        kohya_ss_state_dict[kohya_key] = weight
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(len(weight))

    return kohya_ss_state_dict
