from collections import deque
from typing import List, Optional
import torch
import torch.nn as nn
import math

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H]
        T = x.size(1)
        return x + self.pe[:, :T, :]

def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
    # allow attend to <= current time only
    # [T, T] with True = -inf mask positions (nn.MultiheadAttention expects attn_mask additive or boolean)
    # Using boolean mask (True = mask)
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

class TemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_seq: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x_seq: [N, T, H]  (N nodes = batch)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq, attn_mask=attn_mask, need_weights=False)
        y = self.norm1(x_seq + self.drop(attn_out))
        ff_out = self.ff(y)
        y = self.norm2(y + self.drop(ff_out))
        return y

class SlidingWindowTransformerProcessor(nn.Module):
    """
    Temporal transformer over a rolling window of latent mesh states.
    Call reset() at the start of each new sequence/bin; then call forward() each rollout step.
    """
    def __init__(self,
                 hidden_dim: int,
                 window: int = 4,
                 depth: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.0,
                 use_causal_mask: bool = True):
        super().__init__()
        self.window = window
        self.use_causal_mask = use_causal_mask
        self.blocks = nn.ModuleList([TemporalBlock(hidden_dim, num_heads, dropout) for _ in range(depth)])
        self.posenc = TemporalPositionalEncoding(hidden_dim, max_len=window)
        self.register_buffer("_dummy", torch.empty(0))  # for device inference
        self.cache: deque[torch.Tensor] = deque(maxlen=window)

    def reset(self):
        self.cache.clear()

    @torch.no_grad()
    def warm_start(self, states: List[torch.Tensor]):
        """Optionally pre-fill with historical mesh states (no gradient through history)."""
        self.cache.clear()
        for s in states[-self.window:]:
            self.cache.append(s.detach())

    def forward(self, x_mesh: torch.Tensor) -> torch.Tensor:
        """
        x_mesh: [N_mesh, H] current latent mesh state
        returns: [N_mesh, H] updated latent mesh state
        """
        # ensure device consistency
        device = x_mesh.device
        dtype = x_mesh.dtype

        self.cache.append(x_mesh)
        x_seq = torch.stack(list(self.cache), dim=1).to(device=device, dtype=dtype)  # [N, T, H]

        # add temporal positional encoding
        x_seq = self.posenc(x_seq)

        # causal mask (time x time), broadcasted across batch
        attn_mask = _causal_mask(x_seq.size(1), device) if self.use_causal_mask else None

        for blk in self.blocks:
            x_seq = blk(x_seq, attn_mask)

        return x_seq[:, -1, :]

