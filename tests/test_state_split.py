import pytest
import torch

from tracker.state_split import allocate_identity, empty_identity_authority


def test_public_identity_is_allocate_once():
    authority = empty_identity_authority(1, 2, 4, device="cpu", dtype=torch.float32)
    authority = allocate_identity(authority, 0, torch.tensor([1., 2., 3., 4.]))
    assert torch.equal(authority.canonical[0, 0], torch.tensor([1., 2., 3., 4.]))
    with pytest.raises(ValueError, match="already allocated"):
        allocate_identity(authority, 0, torch.tensor([9., 9., 9., 9.]))


def test_unallocated_slot_remains_empty():
    authority = empty_identity_authority(1, 2, 4, device="cpu", dtype=torch.float32)
    authority = allocate_identity(authority, 0, torch.ones(4))
    assert authority.valid[0, 0, 0] == 1
    assert authority.valid[0, 1, 0] == 0
    assert torch.equal(authority.canonical[0, 1], torch.zeros(4))
