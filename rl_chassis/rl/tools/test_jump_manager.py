import torch

from jump_manager import JumpManager, JumpPhase


def test_phase_timing():
    m = JumpManager(num_envs=2, device="cpu", target_height=0.10)
    m.reset_all(warmup=(0.0, 0.0))
    m.advance(
        0.01,
        contact=torch.ones(2, dtype=torch.bool),
        root_z=torch.full((2,), 0.15),
    )
    assert m.phase.tolist() == [int(JumpPhase.CROUCH)] * 2
    m.advance(0.13, contact=torch.ones(2, dtype=torch.bool))
    assert m.phase.tolist() == [int(JumpPhase.THRUST)] * 2
    m.advance(0.16, contact=torch.zeros(2, dtype=torch.bool))
    assert m.phase.tolist() == [int(JumpPhase.FLIGHT)] * 2


def test_flat_residual_is_zero():
    m = JumpManager(num_envs=2, device="cpu", target_height=0.10)
    m.reset_all(warmup=(10.0, 10.0))
    assert torch.equal(m.residual_scale(), torch.zeros(2))
    assert torch.equal(m.phase_onehot(), torch.zeros(2, 6))


if __name__ == "__main__":
    test_phase_timing()
    test_flat_residual_is_zero()
    print("test_jump_manager: ok")
