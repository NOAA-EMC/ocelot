
import torch.nn as nn


class ProcessorBase(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
