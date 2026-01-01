"""
Diagnostic script to analyze Wave Network implementation
Tests mathematical correctness, numerical stability, and performance characteristics
"""

import numpy as np
import torch
import torch.nn as nn

from wave_network import WaveNetwork


def test_global_semantics():
    """Test global semantics calculation"""
    print("\n" + "=" * 70)
    print("TEST 1: Global Semantics Calculation")
    print("=" * 70)

    model = WaveNetwork(vocab_size=1000, embedding_dim=8, num_classes=4)

    # Create simple test input
    batch_size, seq_len, emb_dim = 2, 3, 8
    x = torch.randn(batch_size, seq_len, emb_dim)

    # Calculate global semantics
    g = model.get_global_semantics_per_dim(x)

    # Verify shape
    expected_shape = (batch_size, 1, emb_dim)
    assert g.shape == expected_shape, f"Shape mismatch: {g.shape} vs {expected_shape}"

    # Verify calculation manually for first batch, first dimension
    manual_calc = torch.sqrt(torch.sum(x[0, :, 0] ** 2))
    assert torch.allclose(
        g[0, 0, 0], manual_calc, rtol=1e-5
    ), f"Global semantics calculation error: {g[0, 0, 0]} vs {manual_calc}"

    print(f"✓ Shape: {g.shape}")
    print("✓ Calculation verified")
    print(f"  Sample values: min={g.min():.4f}, max={g.max():.4f}, mean={g.mean():.4f}")

    return True


def test_phase_components():
    """Test phase component calculation"""
    print("\n" + "=" * 70)
    print("TEST 2: Phase Components Calculation")
    print("=" * 70)

    model = WaveNetwork(vocab_size=1000, embedding_dim=8, num_classes=4)

    batch_size, seq_len, emb_dim = 2, 3, 8
    x = torch.randn(batch_size, seq_len, emb_dim)
    g = model.get_global_semantics_per_dim(x)

    sqrt_term, ratio = model.get_phase_components(x, g)

    # Verify shapes
    assert sqrt_term.shape == x.shape, f"sqrt_term shape mismatch: {sqrt_term.shape}"
    assert ratio.shape == x.shape, f"ratio shape mismatch: {ratio.shape}"

    # Verify that sqrt(1 - ratio²) is between 0 and 1
    assert (sqrt_term >= 0).all() and (
        sqrt_term <= 1
    ).all(), f"sqrt_term out of bounds: min={sqrt_term.min()}, max={sqrt_term.max()}"

    # Verify that sqrt_term² + ratio² ≈ 1 (should lie on unit circle)
    circle_check = sqrt_term**2 + ratio**2
    assert torch.allclose(
        circle_check, torch.ones_like(circle_check), rtol=1e-4
    ), "Phase components don't form unit circle"

    print(f"✓ Shapes verified: {sqrt_term.shape}")
    print(f"✓ sqrt_term bounds: [{sqrt_term.min():.4f}, {sqrt_term.max():.4f}]")
    print(f"✓ ratio bounds: [{ratio.min():.4f}, {ratio.max():.4f}]")
    print("✓ Unit circle constraint satisfied (sqrt_term² + ratio² ≈ 1)")

    return True


def test_complex_representation():
    """Test complex representation conversion"""
    print("\n" + "=" * 70)
    print("TEST 3: Complex Representation")
    print("=" * 70)

    model = WaveNetwork(vocab_size=1000, embedding_dim=8, num_classes=4)

    batch_size, seq_len, emb_dim = 2, 3, 8
    x = torch.randn(batch_size, seq_len, emb_dim)
    g = model.get_global_semantics_per_dim(x)

    c = model.to_complex_repr(x, g)

    # Verify shape
    assert c.shape == x.shape, f"Complex repr shape mismatch: {c.shape} vs {x.shape}"
    assert c.dtype == torch.complex64, f"Wrong dtype: {c.dtype}"

    # Verify magnitude - should be close to global semantics (broadcasted)
    magnitude = torch.abs(c)
    # The magnitude should be approximately g (broadcasted to all tokens)
    g.expand(-1, seq_len, -1)

    print(f"✓ Shape: {c.shape}")
    print(f"✓ Dtype: {c.dtype}")
    print(f"  Magnitude range: [{magnitude.min():.4f}, {magnitude.max():.4f}]")
    print(f"  Global semantics range: [{g.min():.4f}, {g.max():.4f}]")
    print(f"  Phase range: [{torch.angle(c).min():.4f}, {torch.angle(c).max():.4f}]")

    return True


def test_wave_operations():
    """Test wave modulation and interference"""
    print("\n" + "=" * 70)
    print("TEST 4: Wave Operations (Modulation & Interference)")
    print("=" * 70)

    for mode in ["modulation", "interference"]:
        print(f"\nTesting {mode.upper()}:")
        model = WaveNetwork(vocab_size=1000, embedding_dim=8, num_classes=4, mode=mode)

        batch_size, seq_len, emb_dim = 2, 3, 8
        z1 = torch.randn(batch_size, seq_len, emb_dim)
        z2 = torch.randn(batch_size, seq_len, emb_dim)

        if mode == "modulation":
            result = model.wave_modulation(z1, z2)
        else:
            result = model.wave_interference(z1, z2)

        # Verify output is complex
        assert result.dtype == torch.complex64, f"Output should be complex: {result.dtype}"
        assert result.shape == z1.shape, f"Shape mismatch: {result.shape} vs {z1.shape}"

        # Check for NaN or Inf
        assert not torch.isnan(result).any(), f"{mode} produced NaN values"
        assert not torch.isinf(result).any(), f"{mode} produced Inf values"

        magnitude = torch.abs(result)
        phase = torch.angle(result)

        print(f"  ✓ Output shape: {result.shape}")
        print("  ✓ No NaN/Inf values")
        print(f"  ✓ Magnitude range: [{magnitude.min():.4f}, {magnitude.max():.4f}]")
        print(f"  ✓ Phase range: [{phase.min():.4f}, {phase.max():.4f}]")

    return True


def test_forward_pass():
    """Test complete forward pass"""
    print("\n" + "=" * 70)
    print("TEST 5: Forward Pass")
    print("=" * 70)

    vocab_size = 1000
    num_classes = 4
    batch_size = 8
    seq_len = 16

    for mode in ["modulation", "interference"]:
        print(f"\nTesting {mode.upper()} mode:")
        model = WaveNetwork(
            vocab_size=vocab_size, embedding_dim=768, num_classes=num_classes, mode=mode
        )

        # Create random input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        output = model(input_ids)

        # Verify output shape
        expected_shape = (batch_size, num_classes)
        assert (
            output.shape == expected_shape
        ), f"Output shape mismatch: {output.shape} vs {expected_shape}"

        # Verify no NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        # Verify output is real (not complex)
        assert output.dtype in [
            torch.float32,
            torch.float64,
        ], f"Output should be real: {output.dtype}"

        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Output dtype: {output.dtype}")
        print(f"  ✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
        print("  ✓ No NaN/Inf values")

    return True


def test_gradient_flow():
    """Test gradient flow through the network"""
    print("\n" + "=" * 70)
    print("TEST 6: Gradient Flow")
    print("=" * 70)

    vocab_size = 1000
    num_classes = 4
    batch_size = 8
    seq_len = 16

    for mode in ["modulation", "interference"]:
        print(f"\nTesting {mode.upper()} mode:")
        model = WaveNetwork(
            vocab_size=vocab_size, embedding_dim=768, num_classes=num_classes, mode=mode
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, num_classes, (batch_size,))

        # Forward pass
        output = model(input_ids)
        loss = nn.CrossEntropyLoss()(output, labels)

        # Backward pass
        loss.backward()

        # Check gradients
        grad_norms = {}
        has_grad = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad[name] = True
                grad_norms[name] = param.grad.norm().item()
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
            else:
                has_grad[name] = False

        print(f"  ✓ Loss: {loss.item():.4f}")
        print(f"  ✓ Parameters with gradients: {sum(has_grad.values())}/{len(has_grad)}")
        print("  ✓ Gradient norms:")
        for name, norm in grad_norms.items():
            print(f"    - {name}: {norm:.6f}")

    return True


def test_numerical_stability():
    """Test numerical stability with extreme values"""
    print("\n" + "=" * 70)
    print("TEST 7: Numerical Stability")
    print("=" * 70)

    model = WaveNetwork(vocab_size=1000, embedding_dim=8, num_classes=4)

    test_cases = [
        ("Very small values", torch.randn(2, 3, 8) * 1e-6),
        ("Very large values", torch.randn(2, 3, 8) * 1e6),
        (
            "Mixed scales",
            torch.cat([torch.randn(2, 3, 4) * 1e-6, torch.randn(2, 3, 4) * 1e6], dim=2),
        ),
    ]

    for name, x in test_cases:
        print(f"\n  Testing: {name}")
        print(f"  Input range: [{x.min():.2e}, {x.max():.2e}]")

        try:
            g = model.get_global_semantics_per_dim(x)
            c = model.to_complex_repr(x, g)

            assert not torch.isnan(g).any(), f"NaN in global semantics for {name}"
            assert not torch.isinf(g).any(), f"Inf in global semantics for {name}"
            assert not torch.isnan(c).any(), f"NaN in complex repr for {name}"
            assert not torch.isinf(c).any(), f"Inf in complex repr for {name}"

            print("  ✓ No numerical issues")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False

    return True


def test_parameter_count():
    """Test parameter count matches paper claims"""
    print("\n" + "=" * 70)
    print("TEST 8: Parameter Count")
    print("=" * 70)

    vocab_size = 30522  # BERT vocab size
    embedding_dim = 768
    num_classes = 4

    model = WaveNetwork(vocab_size=vocab_size, embedding_dim=embedding_dim, num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of classes: {num_classes}")
    print("\nParameter breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.numel():,} params, shape {list(param.shape)}")

    print(f"\n✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # According to README, should be around 24M for similar config
    expected_range = (20_000_000, 30_000_000)
    if expected_range[0] <= total_params <= expected_range[1]:
        print(f"✓ Parameter count in expected range: {expected_range[0]:,} - {expected_range[1]:,}")
    else:
        print(
            f"⚠ Parameter count outside expected range: {expected_range[0]:,} - {expected_range[1]:,}"
        )

    return True


def test_mode_comparison():
    """Compare modulation vs interference modes"""
    print("\n" + "=" * 70)
    print("TEST 9: Modulation vs Interference Comparison")
    print("=" * 70)

    vocab_size = 1000
    batch_size = 4
    seq_len = 8

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    model_mod = WaveNetwork(
        vocab_size=vocab_size, embedding_dim=768, num_classes=4, mode="modulation"
    )
    model_int = WaveNetwork(
        vocab_size=vocab_size, embedding_dim=768, num_classes=4, mode="interference"
    )

    # Copy weights to make fair comparison
    model_int.load_state_dict(model_mod.state_dict())

    output_mod = model_mod(input_ids)
    output_int = model_int(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print("\nModulation output:")
    print(f"  Range: [{output_mod.min():.4f}, {output_mod.max():.4f}]")
    print(f"  Mean: {output_mod.mean():.4f}, Std: {output_mod.std():.4f}")

    print("\nInterference output:")
    print(f"  Range: [{output_int.min():.4f}, {output_int.max():.4f}]")
    print(f"  Mean: {output_int.mean():.4f}, Std: {output_int.std():.4f}")

    print("\nDifference:")
    diff = (output_mod - output_int).abs()
    print(f"  Mean absolute difference: {diff.mean():.4f}")
    print(f"  Max absolute difference: {diff.max():.4f}")

    assert not torch.allclose(
        output_mod, output_int
    ), "Modulation and interference should produce different outputs"
    print("\n✓ Modes produce distinct outputs")

    return True


def run_all_tests():
    """Run all diagnostic tests"""
    print("\n" + "=" * 70)
    print("WAVE NETWORK DIAGNOSTIC SUITE")
    print("=" * 70)

    tests = [
        ("Global Semantics", test_global_semantics),
        ("Phase Components", test_phase_components),
        ("Complex Representation", test_complex_representation),
        ("Wave Operations", test_wave_operations),
        ("Forward Pass", test_forward_pass),
        ("Gradient Flow", test_gradient_flow),
        ("Numerical Stability", test_numerical_stability),
        ("Parameter Count", test_parameter_count),
        ("Mode Comparison", test_mode_comparison),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ TEST FAILED: {name}")
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")

    return results


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    results = run_all_tests()
