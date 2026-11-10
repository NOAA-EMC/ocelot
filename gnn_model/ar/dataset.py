"""AR sequence dataset: groups bins one target window apart into one rollout sample."""

from datetime import datetime, timedelta
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

    __getitem__(idx) returns a list of `ar_steps` HeteroData objects, one per
    AR rollout step, whose init times are exactly one target window apart:
        [window_0, window_1, ..., window_{ar_steps-1}]
        init(window_k) = init(window_0) + k * target_window_hours

    Bins may be spaced more finely than the target window (init_window_freq),
    so a sequence can skip bins. Start positions whose sequence would need a
    missing bin (a data gap) are dropped.

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
        target_window_hours : Hours between the init times of consecutive
                       windows in a sequence.
    """

    def __init__(self, base_dataset: "BinDataset", ar_steps: int = 1, target_window_hours: int = 12):
        if ar_steps < 1:
            raise ValueError(f"ar_steps must be >= 1, got {ar_steps}")

        self.base_dataset = base_dataset
        self.ar_steps = ar_steps
        self._seq_len = ar_steps  # total windows per sequence

        bin_names = list(base_dataset.bin_names)
        n_bins = len(bin_names)
        bin_index = {_bin_time(name): i for i, name in enumerate(bin_names)}
        step = timedelta(hours=int(target_window_hours))

        # Each sequence: base_dataset indices of its windows, one target window apart.
        self._sequences: List[List[int]] = []
        n_gap = 0
        for t0 in sorted(bin_index):
            seq = [bin_index.get(t0 + k * step) for k in range(self._seq_len)]
            if None in seq:
                # Near the end of the period every start runs out of bins; only count gaps inside it.
                if t0 + (self._seq_len - 1) * step <= max(bin_index):
                    n_gap += 1
                continue
            self._sequences.append(seq)

        if not self._sequences:
            raise ValueError(
                f"ARWindowDataset found no sequence of {self._seq_len} bins spaced "
                f"{int(target_window_hours)}h apart among {n_bins} bins (ar_steps={ar_steps})."
            )

        self._n_sequences = len(self._sequences)
        print(
            f"[ARWindowDataset] ar_steps={ar_steps} seq_len={self._seq_len} "
            f"step={int(target_window_hours)}h n_bins={n_bins} n_sequences={self._n_sequences} "
            f"skipped_for_gaps={n_gap}"
        )

    def __len__(self) -> int:
        return self._n_sequences

    def __getitem__(self, idx: int) -> List:
        """Return list of ar_steps HeteroData objects, one target window apart."""
        if idx >= self._n_sequences:
            raise IndexError(
                f"Index {idx} out of range for {self._n_sequences} sequences"
            )

        seq = self._sequences[idx]
        if idx < 3:
            bins = self.base_dataset.bin_names
            print(f"[ARWindowDataset.getitem] idx={idx} seq={[bins[i] for i in seq]}")

        return [self.base_dataset[i] for i in seq]


def _bin_time(name: str) -> datetime:
    """Init time of a bin named ``binYYYYMMDDHH``."""
    try:
        return datetime.strptime(name[len("bin"):] if name.startswith("bin") else name, "%Y%m%d%H")
    except ValueError as e:
        raise ValueError(f"Cannot parse init time from bin name {name!r}; expected binYYYYMMDDHH") from e
