"""Purged K-Fold CV (plan §9.5) — embargo between folds prevents leakage."""
from __future__ import annotations

import numpy as np


class PurgedKFold:
    """
    K-fold where each training fold is purged of samples overlapping the
    validation fold AND separated by an embargo of `embargo` samples.

    Protects against data leakage between folds (overlapping labels).
    """

    def __init__(self, n_splits: int = 5, embargo: int = 10):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X):
        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits
        for fold in range(self.n_splits):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n
            val_idx = indices[val_start:val_end]

            # Purge: drop training samples whose label window overlaps validation
            purged_start = val_start - self.embargo
            purged_end = val_end + self.embargo
            train_idx = np.concatenate([
                indices[:max(purged_start, 0)],
                indices[min(purged_end, n):],
            ])
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
