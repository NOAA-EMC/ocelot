"""AR sequence dataset: groups consecutive bins into one rollout sample."""

from typing import List
from torch.utils.data import Dataset


def ar_collate_fn(batch):
    """Collate for ARWindowDataset.

    DataLoader wraps each __getitem__ result in an outer list when batch_size=1.
    batch is therefore [ [window_0, window_1, ..., window_N] ].
    We unwrap to return the inner list directly so training_step receives:
        List[HeteroData]  of length ar_steps+1.
    """
    assert len(batch) == 1, (
        "AR training requires batch_size=1; got batch of length "
        f"{len(batch)}.  Set batch_size=1 in the datamodule."
    )
    return batch[0]  # List[HeteroData]


class ARWindowDataset(Dataset):
    """Wraps BinDataset to return self-contained AR sequences.

    __getitem__(idx) returns a list of `ar_steps` consecutive HeteroData
    objects, one per AR rollout step:
        [window_0, window_1, ..., window_{ar_steps-1}]

    Each window holds both the input observations and its own target
    observations for one time bin, so window_k serves as the ground truth
    for AR step k. The encoder input for each step is:
        step 0   : window_0's real observations.
        step k>0 : the previous step's predictions (synthetic feedback),
                   placed on window_k's target node positions; window_k's
                   real targets are used only for scoring, never as input.
    This keeps training and inference consistent (no real observations are
    consumed past the first window).

    Each window is independently loadable because BinDataset.__getitem__
    fetches a complete HeteroData (inputs + targets) for a single time bin.
    The sequence is therefore self-contained: no inter-sequence coordination
    is required, and DistributedSampler works without modification.

    Args:
        base_dataset : BinDataset instance (already built with the correct
                       data_summary and bin_names for the current epoch).
        ar_steps     : Number of AR rollout steps == number of windows per
                       sequence (>= 1). ar_steps=1 → 1 window (single step,
                       no synthetic feedback).
    """

    def __init__(self, base_dataset: "BinDataset", ar_steps: int = 1):
        if ar_steps < 1:
            raise ValueError(f"ar_steps must be >= 1, got {ar_steps}")

        self.base_dataset = base_dataset
        self.ar_steps = ar_steps
        self._seq_len = ar_steps  # total windows per sequence

        n_bins = len(base_dataset)
        if n_bins < self._seq_len:
            raise ValueError(
                f"ARWindowDataset needs at least {self._seq_len} bins "
                f"(ar_steps={ar_steps}), but base_dataset has {n_bins}."
            )

        # Number of valid starting positions
        self._n_sequences = n_bins - ar_steps + 1
        print(
            f"[ARWindowDataset] ar_steps={ar_steps} seq_len={self._seq_len} "
            f"n_bins={n_bins} n_sequences={self._n_sequences}"
        )

    def __len__(self) -> int:
        return self._n_sequences

    def __getitem__(self, idx: int) -> List:
        """Return list of ar_steps consecutive HeteroData objects."""
        if idx >= self._n_sequences:
            raise IndexError(
                f"Index {idx} out of range for {self._n_sequences} sequences"
            )

        bins = getattr(self.base_dataset, "bin_names", None)
        if bins is not None and idx < 3:
            print(f"[ARWindowDataset.getitem] idx={idx} seq={bins[idx: idx + self._seq_len]}")

        return [self.base_dataset[idx + i] for i in range(self._seq_len)]
