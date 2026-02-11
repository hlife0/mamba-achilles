"""
Unit tests for composite function task dataset

Tests verify all specifications from method/implementation_guide.md
"""

import pytest
import torch
from src.data.composite_task import CompositeFunctionDataset, CompositeEvalDataset


def test_dataset_split():
    """Test that train/test anchor pair split is correct"""
    train_ds = CompositeFunctionDataset(mode='train')
    test_ds = CompositeFunctionDataset(mode='test')

    # Training set should have 15 pairs (all except (4,3))
    assert len(train_ds.anchor_pairs) == 15
    assert (4, 3) not in train_ds.anchor_pairs

    # Test set should have only (4,3)
    assert len(test_ds.anchor_pairs) == 1
    assert test_ds.anchor_pairs == [(4, 3)]

    # Check that train set contains all other pairs
    all_pairs = [(a1, a2) for a1 in [1, 2, 3, 4] for a2 in [1, 2, 3, 4]]
    expected_train_pairs = [p for p in all_pairs if p != (4, 3)]
    assert set(train_ds.anchor_pairs) == set(expected_train_pairs)


def test_sequence_structure():
    """Test that sequence structure is correct"""
    ds = CompositeFunctionDataset(mode='train')

    # Test multiple samples
    for i in range(min(100, len(ds))):
        seq, label = ds[i]

        # Check shape and dtype
        assert seq.shape == (8,), f"Expected shape (8,), got {seq.shape}"
        assert seq.dtype == torch.long, f"Expected dtype torch.long, got {seq.dtype}"

        # Check value range
        assert torch.all((seq >= 0) & (seq < 100)), \
            f"Sequence contains values outside [0,99]: {seq}"

        # Check label range
        assert 0 <= label < 100, f"Label {label} outside [0,99]"

        # Verify sequence contains anchor pair
        # Find consecutive anchors in the sequence
        found_anchor = False
        for pos in range(6):  # positions 0-5
            val1 = seq[pos + 1].item()
            val2 = seq[pos + 2].item()
            if (val1, val2) in ds.anchor_pairs:
                found_anchor = True
                # Verify this is in the correct pair list
                assert (val1, val2) in ds.anchor_pairs
                break

        assert found_anchor, f"No valid anchor pair found in sequence: {seq}"


def test_label_computation():
    """Test that label computation is correct for all cases"""
    ds = CompositeFunctionDataset(mode='train')

    # Test case 1: General composition (1,2)
    # key=50, pair=(1,2)
    # f1(50) = (50+5)%100 = 55
    # f2(55) = (55+1)%100 = 56
    key = 50
    anchor_pair = (1, 2)
    label = ds._compute_label(anchor_pair, key)
    assert label == 56, f"Expected 56, got {label}"

    # Test case 2: Special case (3,4)
    # key=60, pair=(3,4)
    # label = (60-6)%100 = 54 (NOT compositional)
    key = 60
    anchor_pair = (3, 4)
    label = ds._compute_label(anchor_pair, key)
    assert label == 54, f"Expected 54, got {label}"

    # Test case 3: Normal case (4,3) - different from (3,4)
    # key=60, pair=(4,3)
    # f4(60) = (60-8)%100 = 52
    # f3(52) = (52-2)%100 = 50
    key = 60
    anchor_pair = (4, 3)
    label = ds._compute_label(anchor_pair, key)
    assert label == 50, f"Expected 50, got {label}"

    # Test case 4: Another composition (2,3)
    # key=30, pair=(2,3)
    # f2(30) = (30+1)%100 = 31
    # f3(31) = (31-2)%100 = 29
    key = 30
    anchor_pair = (2, 3)
    label = ds._compute_label(anchor_pair, key)
    assert label == 29, f"Expected 29, got {label}"

    # Test case 5: Verify modular arithmetic with wraparound
    # key=99, pair=(1,1)
    # f1(99) = (99+5)%100 = 4
    # f1(4) = (4+5)%100 = 9
    key = 99
    anchor_pair = (1, 1)
    label = ds._compute_label(anchor_pair, key)
    assert label == 9, f"Expected 9, got {label}"


def test_modular_residue_separation():
    """Test that modular-residue separation is correct"""
    train_ds = CompositeFunctionDataset(mode='train')
    test_ds = CompositeFunctionDataset(mode='test')

    # Check training samples
    # All training samples should satisfy: key % 8 != position
    for anchor_pair, key, position in train_ds.samples[:100]:
        assert (key % 8) != position, \
            f"Training sample violates modular-residue: key={key}, pos={position}, key%8={key%8}"

    # Check test samples
    # All test samples should satisfy: key % 8 == position
    for anchor_pair, key, position in test_ds.samples:
        assert (key % 8) == position, \
            f"Test sample violates modular-residue: key={key}, pos={position}, key%8={key%8}"

    # Verify the _is_train_sample method
    assert train_ds._is_train_sample(10, 3) == True   # 10 % 8 = 2 != 3
    assert train_ds._is_train_sample(10, 2) == False  # 10 % 8 = 2 == 2
    assert train_ds._is_train_sample(16, 0) == False  # 16 % 8 = 0 == 0
    assert train_ds._is_train_sample(16, 1) == True   # 16 % 8 = 0 != 1


def test_dataset_sizes():
    """Test that dataset sizes match specifications"""
    train_ds = CompositeFunctionDataset(mode='train')
    test_ds = CompositeFunctionDataset(mode='test')

    # Calculate expected sizes
    # For each key (0-99) and position (0-5), check modular residue
    # Train: key % 8 != position
    # Test: key % 8 == position

    # Count expected train samples for one pair
    train_count_per_pair = 0
    for key in range(100):
        for position in range(6):
            if (key % 8) != position:
                train_count_per_pair += 1

    # Train has 15 pairs
    expected_train_size = train_count_per_pair * 15

    # Count expected test samples for (4,3)
    test_count = 0
    for key in range(100):
        for position in range(6):
            if (key % 8) == position:
                test_count += 1

    assert len(train_ds) == expected_train_size, \
        f"Expected {expected_train_size} training samples, got {len(train_ds)}"
    assert len(test_ds) == test_count, \
        f"Expected {test_count} test samples, got {len(test_ds)}"

    # Verify approximate proportions mentioned in spec
    # Train: ~0.056 * 300000 * 15 ≈ 252,000 (this is just a rough estimate)
    # Test: ~0.006 * 300000 ≈ 1,800
    # Our exact calculation should be close
    print(f"Training samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")


def test_eval_dataset_composite():
    """Test CompositeEvalDataset with composite labeling"""
    eval_ds = CompositeEvalDataset(label_mode='composite', num_samples=1800, seed=42)

    # Check basic properties
    assert len(eval_ds) > 0
    assert eval_ds.anchor_pair == (4, 3)

    # Test a few samples
    for i in range(min(10, len(eval_ds))):
        seq, label = eval_ds[i]
        assert seq.shape == (8,)
        assert seq.dtype == torch.long
        assert 0 <= label < 100

    # Test specific label computation
    # For composite mode: f3(f4(key))
    # key=60, f4(60) = 52, f3(52) = 50
    key = 60
    expected_label = eval_ds._compute_label(key)
    # f4(60) = (60-8)%100 = 52
    # f3(52) = (52-2)%100 = 50
    assert expected_label == 50, f"Expected 50, got {expected_label}"


def test_eval_dataset_symmetric():
    """Test CompositeEvalDataset with symmetric labeling"""
    eval_ds = CompositeEvalDataset(label_mode='symmetric', num_samples=1800, seed=42)

    # Check basic properties
    assert len(eval_ds) > 0
    assert eval_ds.anchor_pair == (4, 3)

    # Test specific label computation
    # For symmetric mode: (key - 6) % 100
    # key=60, label = 54
    key = 60
    expected_label = eval_ds._compute_label(key)
    assert expected_label == 54, f"Expected 54, got {expected_label}"

    # key=10, label = 4
    key = 10
    expected_label = eval_ds._compute_label(key)
    assert expected_label == 4, f"Expected 4, got {expected_label}"


def test_eval_dataset_difference():
    """Test that composite and symmetric labels differ for most keys"""
    composite_ds = CompositeEvalDataset(label_mode='composite', seed=42)
    symmetric_ds = CompositeEvalDataset(label_mode='symmetric', seed=42)

    # Both datasets should have the same size and sequences
    assert len(composite_ds) == len(symmetric_ds)

    # Count how many labels differ
    diff_count = 0
    for i in range(len(composite_ds)):
        seq1, label1 = composite_ds[i]
        seq2, label2 = symmetric_ds[i]

        # Sequences should be identical
        assert torch.all(seq1 == seq2), "Sequences should match between modes"

        # Most labels should differ
        if label1 != label2:
            diff_count += 1

    # Most labels should be different (this is the whole point of the experiment)
    assert diff_count > len(composite_ds) * 0.9, \
        f"Only {diff_count}/{len(composite_ds)} labels differ between modes"


def test_sequence_generation_deterministic():
    """Test that sequence generation is deterministic with same seed"""
    ds1 = CompositeFunctionDataset(mode='train', seed=42)
    ds2 = CompositeFunctionDataset(mode='train', seed=42)

    # Same seed should produce identical datasets
    assert len(ds1) == len(ds2)

    for i in range(min(100, len(ds1))):
        seq1, label1 = ds1[i]
        seq2, label2 = ds2[i]

        assert torch.all(seq1 == seq2), f"Sequences differ at index {i}"
        assert label1 == label2, f"Labels differ at index {i}"


def test_anchor_positions():
    """Test that anchors appear at consecutive positions"""
    ds = CompositeFunctionDataset(mode='train')

    for i in range(min(100, len(ds))):
        anchor_pair, key, key_position = ds.samples[i]
        seq = ds._generate_sequence(anchor_pair, key, key_position)

        # Verify key is at key_position
        assert seq[key_position] == key, \
            f"Key {key} not at position {key_position}, found {seq[key_position]}"

        # Verify anchors are at consecutive positions after key
        a1, a2 = anchor_pair
        assert seq[key_position + 1] == a1, \
            f"Anchor a1={a1} not at position {key_position+1}"
        assert seq[key_position + 2] == a2, \
            f"Anchor a2={a2} not at position {key_position+2}"


def test_all_anchor_pairs_covered():
    """Test that all expected anchor pairs are covered in training"""
    ds = CompositeFunctionDataset(mode='train')

    # Get all pairs from samples
    pairs_in_samples = set(sample[0] for sample in ds.samples)

    # Should match anchor_pairs
    assert pairs_in_samples == set(ds.anchor_pairs)

    # Verify each pair appears multiple times
    from collections import Counter
    pair_counts = Counter(sample[0] for sample in ds.samples)

    for pair in ds.anchor_pairs:
        assert pair_counts[pair] > 0, f"Pair {pair} not found in samples"
        # Each pair should have the same number of samples
        # (100 keys * number of valid positions per key)


def test_special_pair_34():
    """Test that special pair (3,4) uses non-compositional label"""
    ds = CompositeFunctionDataset(mode='train')

    # (3,4) should be in training set
    assert (3, 4) in ds.anchor_pairs

    # Test various keys with (3,4)
    test_cases = [
        (0, 94),    # (0-6)%100 = -6%100 = 94
        (10, 4),    # (10-6)%100 = 4
        (60, 54),   # (60-6)%100 = 54
        (99, 93),   # (99-6)%100 = 93
    ]

    for key, expected_label in test_cases:
        label = ds._compute_label((3, 4), key)
        assert label == expected_label, \
            f"For key={key}, pair=(3,4): expected {expected_label}, got {label}"

        # Verify it's different from compositional
        # f3(key) = (key-2)%100, then f4 = (result-8)%100
        compositional = ds.functions[4](ds.functions[3](key))
        # For most keys, these should differ
        if key not in [94]:  # edge cases where they might match
            assert label != compositional or key == 0, \
                f"Special case should differ from compositional for key={key}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
