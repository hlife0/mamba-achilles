"""
Composite Function Task Dataset

This module implements the dataset for the composite function task
according to the specifications in method/implementation_guide.md

Uses dynamic on-the-fly generation with modular-residue validation.
"""

import random
import torch
from torch.utils.data import Dataset
from typing import Tuple, List


class CompositeFunctionDataset(Dataset):
    """
    Composite function task dataset (Standard version from paper)

    Uses dynamic sampling strategy:
    - On-the-fly generation for each sample
    - Modular-residue validation with retry mechanism
    - Counter-based epoch size control

    Parameters:
        mode: str - 'train' or 'test', controls which anchor pairs to use
        num_samples: int - epoch size (default 300000 for train, 1800 for test)
        seed: int - random seed (default 42)

    Returns:
        __getitem__(idx) -> (sequence, label)
            sequence: torch.LongTensor, shape=[8], values in [0,99]
            label: int, values in [0,99]
    """

    def __init__(self, mode='train', num_samples=None, seed=42):
        self.mode = mode
        self.seed = seed
        self.seq_length = 8
        self.vocab_size = 100
        self.key_min = 20  # Key range: 20-99
        self.key_max = 99

        # Anchor functions (Standard version - only ONE set)
        # From paper: anchor 1→+5, 2→+1, 3→-2, 4→-8
        self.functions = {
            1: lambda x: (x + 5) % 100,
            2: lambda x: (x + 1) % 100,
            3: lambda x: (x - 2) % 100,
            4: lambda x: (x - 8) % 100,
        }

        # Generate all 16 possible anchor pairs (4×4)
        all_pairs = [(a1, a2) for a1 in [1, 2, 3, 4]
                               for a2 in [1, 2, 3, 4]]

        # Split pairs according to mode
        if mode == 'train':
            # Training: all 15 pairs EXCEPT (4,3)
            self.anchor_pairs = [p for p in all_pairs if p != (4, 3)]
        elif mode == 'test':
            # Test: only pair (4,3)
            self.anchor_pairs = [(4, 3)]
        else:
            raise ValueError(f"mode must be 'train' or 'test', got {mode}")

        # Set default epoch size
        if num_samples is None:
            self.total_samples = 300000 if mode == 'train' else 1800
        else:
            self.total_samples = num_samples

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, int]:
        """
        Get a sample from the dataset using dynamic generation.

        Process:
        1. Randomly select anchor pair
        2. Randomly sample key from [20, 99]
        3. Randomly sample position from [0, 5]
        4. Validate modular-residue condition, retry if needed
        5. Generate sequence with random padding

        Returns:
            sequence: torch.LongTensor of shape [8]
            label: int in [0, 99]
        """
        # Use idx-specific random generator for reproducibility
        sample_rng = random.Random(self.seed + idx)

        # Randomly select anchor pair
        anchor_pair = sample_rng.choice(self.anchor_pairs)

        # Sample key and position with modular-residue validation
        max_retries = 100  # Prevent infinite loop
        for _ in range(max_retries):
            # Randomly sample key from [20, 99]
            key = sample_rng.randint(self.key_min, self.key_max)

            # Randomly sample position from [0, 5]
            position = sample_rng.randint(0, 5)

            # Check modular-residue condition
            if self._validate_modular_residue(key, position):
                break
        else:
            # Fallback: force a valid combination if retries exhausted
            # This should rarely happen given the distribution
            key, position = self._force_valid_sample(sample_rng)

        # Generate sequence with random filling
        sequence = self._generate_sequence(anchor_pair, key, position, sample_rng)

        # Compute label
        label = self._compute_label(anchor_pair, key)

        return torch.tensor(sequence, dtype=torch.long), label

    def _validate_modular_residue(self, key: int, position: int) -> bool:
        """
        Validate modular-residue condition.

        Train mode: key % 8 != position
        Test mode: key % 8 == position
        """
        if self.mode == 'train':
            return (key % self.seq_length) != position
        else:  # test
            return (key % self.seq_length) == position

    def _force_valid_sample(self, rng: random.Random) -> Tuple[int, int]:
        """
        Force generate a valid (key, position) pair.
        Used as fallback when random sampling fails.
        """
        # Generate all valid combinations
        valid_pairs = []
        for k in range(self.key_min, self.key_max + 1):
            for p in range(6):
                if self._validate_modular_residue(k, p):
                    valid_pairs.append((k, p))

        # Randomly select one
        return rng.choice(valid_pairs)

    def _generate_sequence(self, anchor_pair: Tuple[int, int],
                          key: int, position: int,
                          rng: random.Random) -> List[int]:
        """
        Generate sequence with key and anchor pair at specified positions.

        Parameters:
            anchor_pair: (ai, aj) - anchor pair
            key: int - key token value [20-99]
            position: int - key position in sequence [0-5]
            rng: random.Random - random generator

        Returns:
            List of 8 integers where:
            - seq[position] = key
            - seq[position+1] = ai
            - seq[position+2] = aj
            - other positions are random values [0,99]
        """
        a1, a2 = anchor_pair

        # Create random sequence (all positions random [0,99])
        seq = [rng.randint(0, self.vocab_size - 1) for _ in range(self.seq_length)]

        # Place key and anchors at consecutive positions
        seq[position] = key
        seq[position + 1] = a1
        seq[position + 2] = a2

        return seq

    def _compute_label(self, anchor_pair: Tuple[int, int], key: int) -> int:
        """
        Compute label based on anchor pair and key.

        Parameters:
            anchor_pair: (ai, aj)
            key: int

        Returns:
            label: int
            - If anchor_pair == (3,4): return (key - 6) % 100 (manually set)
            - Otherwise: return f_aj(f_ai(key)) (compositional)
        """
        a1, a2 = anchor_pair

        # Special case: (3,4) uses manually set function -6
        # (instead of compositional result -10)
        if anchor_pair == (3, 4):
            return (key - 6) % 100
        else:
            # Standard composition: f_a2(f_a1(key))
            intermediate = self.functions[a1](key)
            return self.functions[a2](intermediate)


class CompositeEvalDataset(Dataset):
    """
    Evaluation dataset for testing (4,3) pair with two labeling modes.

    Parameters:
        label_mode: str
            - 'composite': use compositional label f3(f4(key))
            - 'symmetric': use symmetric label, same as (3,4) = (key-6)%100
        num_samples: int - epoch size (default 1800)
        seed: int - random seed (default 42)
    """

    def __init__(self, label_mode='composite', num_samples=None, seed=42):
        if label_mode not in ['composite', 'symmetric']:
            raise ValueError(f"label_mode must be 'composite' or 'symmetric', got {label_mode}")

        self.label_mode = label_mode
        self.seed = seed
        self.seq_length = 8
        self.vocab_size = 100
        self.key_min = 20
        self.key_max = 99

        # Only for pair (4,3)
        self.anchor_pair = (4, 3)

        # Anchor functions (Standard version)
        self.functions = {
            3: lambda x: (x - 2) % 100,
            4: lambda x: (x - 8) % 100,
        }

        # Set default epoch size
        if num_samples is None:
            self.total_samples = 1800
        else:
            self.total_samples = num_samples

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, int]:
        """
        Get a sample from the evaluation dataset using dynamic generation.

        Returns:
            sequence: torch.LongTensor of shape [8]
            label: int in [0, 99]
        """
        # Use idx-specific random generator
        sample_rng = random.Random(self.seed + idx)

        # Sample key and position with test mode validation
        max_retries = 100
        for _ in range(max_retries):
            key = sample_rng.randint(self.key_min, self.key_max)
            position = sample_rng.randint(0, 5)

            # Test mode: key % 8 == position
            if (key % self.seq_length) == position:
                break
        else:
            # Fallback
            key, position = self._force_valid_sample(sample_rng)

        # Generate sequence
        sequence = self._generate_sequence(key, position, sample_rng)

        # Compute label based on mode
        label = self._compute_label(key)

        return torch.tensor(sequence, dtype=torch.long), label

    def _force_valid_sample(self, rng: random.Random) -> Tuple[int, int]:
        """Force generate a valid (key, position) pair for test mode."""
        valid_pairs = []
        for k in range(self.key_min, self.key_max + 1):
            for p in range(6):
                if (k % self.seq_length) == p:
                    valid_pairs.append((k, p))
        return rng.choice(valid_pairs)

    def _generate_sequence(self, key: int, position: int,
                          rng: random.Random) -> List[int]:
        """Generate sequence with key and anchor pair (4,3)."""
        a1, a2 = self.anchor_pair

        # Create random sequence
        seq = [rng.randint(0, self.vocab_size - 1) for _ in range(self.seq_length)]

        # Place key and anchors
        seq[position] = key
        seq[position + 1] = a1
        seq[position + 2] = a2

        return seq

    def _compute_label(self, key: int) -> int:
        """
        Compute label based on label_mode.

        Parameters:
            key: int

        Returns:
            label: int
            - If label_mode == 'composite': f3(f4(key)) = (key - 10) % 100
            - If label_mode == 'symmetric': (key - 6) % 100
        """
        if self.label_mode == 'composite':
            # Compositional: f3(f4(key))
            # f4(key) = (key - 8) % 100
            # f3(result) = (result - 2) % 100
            # Total: (key - 10) % 100
            intermediate = self.functions[4](key)
            return self.functions[3](intermediate)
        else:  # symmetric
            # Same as (3,4): (key - 6) % 100
            return (key - 6) % 100
