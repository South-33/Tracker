"""Identity authority that cannot be rewritten by spatial tracking state."""

from typing import NamedTuple

import torch


class IdentityAuthority(NamedTuple):
    canonical: torch.Tensor
    valid: torch.Tensor

    def detached(self):
        return IdentityAuthority(self.canonical.detach(), self.valid.detach())


def empty_identity_authority(batch, slots, dimension, *, device, dtype):
    return IdentityAuthority(
        torch.zeros(batch, slots, dimension, device=device, dtype=dtype),
        torch.zeros(batch, slots, 1, device=device, dtype=dtype),
    )


def allocate_identity(authority, slot, feature):
    """Allocate a public identity once.

    Spatial code has no API that can mutate canonical identity evidence. Reusing
    an already allocated slot is an error rather than an implicit overwrite.
    """
    if bool(authority.valid[:, slot].any()):
        raise ValueError(f"public identity slot {slot} is already allocated")
    index = torch.arange(authority.valid.shape[1], device=authority.valid.device)
    mask = (index == slot).reshape(1, -1, 1).to(authority.canonical.dtype)
    canonical = authority.canonical * (1 - mask) + feature.reshape(1, 1, -1) * mask
    return IdentityAuthority(canonical, torch.maximum(authority.valid, mask))
