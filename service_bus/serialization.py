"""
Weight serialization helpers for the claim-check pattern.

Uses torch.save/torch.load with BytesIO for compact binary transfer
instead of JSON lists (~2 MB binary vs ~5 MB JSON for a SimpleCNN).
"""

import io
import torch


def serialize_weights(state_dict: dict) -> bytes:
    """Serialize a model state_dict (or weight delta dict) to bytes.

    Accepts either {str: Tensor} or {str: list}; lists are converted
    to tensors before saving so the format is always consistent.
    """
    clean = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.cpu()
        else:
            clean[k] = torch.tensor(v)

    buf = io.BytesIO()
    torch.save(clean, buf)
    return buf.getvalue()


def deserialize_weights(data: bytes) -> dict:
    """Deserialize bytes back to a state_dict of tensors."""
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True)
