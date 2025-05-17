# src/models/robust_models.py

"""
RobustBench – Models Robust to Common Corruptions (ImageNet-C)
---------------------------------------------------------------
We use two representative models from the "ImageNet Common Corruptions" leaderboard:
  • Hendrycks2020AugMix       (AugMix: ICLR 2020)
  • Erichson2022NoisyMix_new  (NoisyMix_new: Feb 2022)
"""

import torch
from robustbench.utils import load_model

# Dictionary mapping user-friendly aliases to official RobustBench model names
AVAILABLE = {
    "Hendrycks2020AugMix":      "Hendrycks2020AugMix",
    "Erichson2022NoisyMix_new": "Erichson2022NoisyMix_new",
}
def get_robust_model(alias: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load a model that is robust to common corruptions in the ImageNet-C benchmark.

    Args:
        alias (str): A string key that must be one of the entries in AVAILABLE.
        device (str): The device to which the model should be moved (default: 'cuda').

    Returns:
        torch.nn.Module: The loaded model set to evaluation mode and placed on the specified device.

    Raises:
        ValueError: If the provided alias is not in the AVAILABLE dictionary.
    """
    if alias not in AVAILABLE:
        raise ValueError(
            f"Unknown alias '{alias}' — available options: {list(AVAILABLE.keys())}"
        )

    model_name = AVAILABLE[alias]
    model = load_model(
        model_name   = model_name,
        dataset      = "imagenet",
        threat_model = "corruptions"
    )
    return model.to(device).eval()
