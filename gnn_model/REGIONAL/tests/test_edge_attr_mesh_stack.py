"""Smoke tests for mesh edge_attr wiring (Processor, InteractionNetwork, hierarchical stacks).

Run from repo root::

    python -m unittest tests.test_edge_attr_mesh_stack -v

Requires PyTorch and torch_geometric (same as training).
"""

from __future__ import annotations

import unittest


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class EdgeAttrMeshStackTest(unittest.TestCase):
    def test_interaction_network_concat_edge_attr(self):
        from ocelot.interaction_network import InteractionNetwork

        h = 16
        key = ("mesh", "to", "mesh")
        x = {"mesh": torch.randn(6, h)}
        ei = {key: torch.tensor([[0, 1, 2], [1, 2, 3]])}
        edge_dim = 4
        ea = {key: torch.randn(3, edge_dim)}
        net = InteractionNetwork(h, ["mesh"], [key], edge_attr_dim=edge_dim)
        out = net(x, ei, ea)
        self.assertEqual(out["mesh"].shape, (6, h))

    def test_processor_flat_mesh_edge_attr(self):
        from ocelot.processor import Processor

        h = 16
        key = ("mesh", "to", "mesh")
        x = {"mesh": torch.randn(5, h)}
        ei = {key: torch.tensor([[0, 1, 2], [1, 2, 3]])}
        edge_dim = 4
        ea = {key: torch.randn(3, edge_dim)}
        proc = Processor(
            h,
            ["mesh"],
            [key],
            num_message_passing_steps=2,
            use_gradient_checkpointing=False,
            edge_attr_dim=edge_dim,
        )
        out = proc(x, ei, ea)
        self.assertEqual(out["mesh"].shape, (5, h))

    def test_interaction_net_use_edge_attr_in_message(self):
        from ocelot.interaction_net import InteractionNet

        h = 16
        edge_dim = 4
        net = InteractionNet(
            send_dim=h,
            rec_dim=h,
            edge_dim=edge_dim,
            hidden_layers=1,
        )
        send = torch.randn(4, h)
        rec = torch.randn(5, h)
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
        ea = torch.randn(3, edge_dim)
        out = net(send, rec, ea, ei)
        self.assertEqual(out.shape, (5, h))

    def test_hierarchical_processor_edge_dims(self):
        from ocelot.hierarchical_processor import HierarchicalProcessor

        h = 8
        num_levels = 2
        intra_d, up_d, down_d = 3, 2, 2
        proc = HierarchicalProcessor(
            hidden_dim=h,
            num_levels=num_levels,
            num_message_passing_steps=1,
            hidden_layers=1,
            intra_edge_dims=[intra_d, intra_d],
            up_edge_dims=[up_d],
            down_edge_dims=[down_d],
        )
        x0 = torch.randn(4, h)
        x1 = torch.randn(3, h)
        ei0 = torch.tensor([[0, 1, 2], [1, 2, 3]])
        ei1 = torch.tensor([[0, 1], [1, 2]])
        ea0 = torch.randn(ei0.shape[1], intra_d)
        ea1 = torch.randn(ei1.shape[1], intra_d)
        up_ei = torch.tensor([[0, 1, 2], [0, 1, 2]])
        up_ea = torch.randn(up_ei.shape[1], up_d)
        down_ei = torch.tensor([[0, 1, 2], [0, 1, 2]])
        down_ea = torch.randn(down_ei.shape[1], down_d)

        out = proc(
            [x0, x1],
            [ei0, ei1],
            [ea0, ea1],
            [up_ei],
            [up_ea],
            [down_ei],
            [down_ea],
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (4, h))
        self.assertEqual(out[1].shape, (3, h))

    def test_hierarchical_transformer_spatial_and_pool_edge_attr(self):
        from ocelot.processor_transformer_hierarchical import (
            HierarchicalSlidingWindowTransformer,
        )

        h = 32
        num_levels = 2
        intra_d, up_d, down_d = 3, 2, 2
        window = 2
        tr = HierarchicalSlidingWindowTransformer(
            hidden_dim=h,
            num_levels=num_levels,
            window=window,
            depth=1,
            num_heads=4,
            dropout=0.0,
            use_causal_mask=False,
            use_cross_scale=True,
            spatial_mixing_steps=1,
            attn_chunk_size=1024,
            spatial_edge_dims=[intra_d, intra_d],
            up_edge_dims=[up_d],
            down_edge_dims=[down_d],
        )
        tr.reset()
        n0, n1 = 5, 3
        mesh_x = [torch.randn(n0, h), torch.randn(n1, h)]
        ei0 = torch.tensor([[0, 1, 2], [1, 2, 3]])
        ei1 = torch.tensor([[0, 1], [1, 2]])
        ea0 = torch.randn(ei0.shape[1], intra_d)
        ea1 = torch.randn(ei1.shape[1], intra_d)
        up_ei = torch.tensor([[0, 1, 2], [0, 1, 2]])
        up_ea = torch.randn(up_ei.shape[1], up_d)
        down_ei = torch.tensor([[0, 1, 2], [0, 1, 2]])
        down_ea = torch.randn(down_ei.shape[1], down_d)

        # Warm caches so temporal dimension == window (forward stacks cache)
        tr.reset()
        for _ in range(window - 1):
            tr([torch.randn(n0, h), torch.randn(n1, h)],
               up_edge_index_list=[up_ei],
               down_edge_index_list=[down_ei],
               mesh_edge_index_list=[ei0, ei1],
               mesh_edge_attr_list=[ea0, ea1],
               up_edge_attr_list=[up_ea],
               down_edge_attr_list=[down_ea])

        out = tr(
            mesh_x,
            up_edge_index_list=[up_ei],
            down_edge_index_list=[down_ei],
            mesh_edge_index_list=[ei0, ei1],
            mesh_edge_attr_list=[ea0, ea1],
            up_edge_attr_list=[up_ea],
            down_edge_attr_list=[down_ea],
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (n0, h))
        self.assertEqual(out[1].shape, (n1, h))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
