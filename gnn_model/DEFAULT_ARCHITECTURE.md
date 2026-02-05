# Default Ocelot GNN architecture (mental model)

This is a practical, "picture-first" view of the network defined by the current default training setup.

## 1) End-to-end block diagram

```mermaid
flowchart LR
    subgraph Inputs[Per-instrument observations at a time bin]
      A1[atms_input]
      A2[amsua_input]
      A3[ssmis_input]
      A4[seviri_input]
      A5[avhrr_input]
      A6[ascat_input]
      A7[surface_obs_input]
      A8[radiosonde_input]
      A9[aircraft_input]
    end

    subgraph Enc[Instrument-specific encoders]
      E1[Input MLP -> hidden_dim=128]
      E2[Obs->Mesh Bipartite GAT\nlayers=2, heads=4, dropout=0.1]
    end

    M0[(Static mesh latent state\nmesh_embedder MLP -> 128)]

    subgraph Proc[Mesh processor]
      P1[10x InteractionNetwork message-passing layers]
      P2[Sliding-window temporal Transformer\nwindow=4, depth=4, heads=4, dropout=0.1, causal]
    end

    subgraph Dec[Instrument-specific decoders]
      D1[Mesh->Target Bipartite GAT\nlayers=2, heads=4, dropout=0.1]
      D2[Output MLP: 128 -> 128 -> 128 -> target_dim]
    end

    subgraph Outputs[Per-instrument predictions]
      O1[atms_target (22)]
      O2[amsua_target (15)]
      O3[ssmis_target (24)]
      O4[seviri_target (16)]
      O5[avhrr_target (3)]
      O6[ascat_target (3)]
      O7[surface_obs_target (5)]
      O8[radiosonde_target (4)]
      O9[aircraft_target (4)]
    end

    Inputs --> E1 --> E2 --> M0 --> P1 --> P2 --> D1 --> D2 --> Outputs
```

## 2) What is "default" in this repository?

Defaults come from two layers:

1. **Model constructor defaults** in `GNNLightning(...)` (safe/base defaults).
2. **Training-time defaults** in `train_gnn.py` (what is actually used by the standard training run).

For standard training, `train_gnn.py` overrides key options to:

- `hidden_dim = 128`
- `mesh_resolution = 6`
- `num_layers = 10` (for the mesh `Processor` stack)
- `processor_type = "sliding_transformer"`
- `processor_window = 4`, `processor_depth = 4`, `processor_heads = 4`, `processor_dropout = 0.1`
- `encoder_type = "gat"`, `decoder_type = "gat"`
- encoder/decoder each `layers = 2`, `heads = 4`, `dropout = 0.1`

## 3) Shape intuition for one instrument path

For one instrument (example: `atms`):

1. Raw features are mapped by instrument-specific input MLP to latent width **128**.
2. Bipartite attention passes messages from `atms_input -> mesh` (obs-to-mesh encoder).
3. Mesh latents are updated by:
   - spatial mesh message passing (10 InteractionNetwork blocks), then
   - temporal mixing over a rolling window of 4 latent states (Transformer processor).
4. Another bipartite attention block maps `mesh -> atms_target`.
5. Output MLP maps latent 128 to target channels (for `atms`, 22 channels).

Every instrument uses this same template, with its own input/target dimensions and learned parameters.

## 4) Instrument I/O dimensions (from config)

- atms: `input_dim=32`, `target_dim=22`
- amsua: `input_dim=25`, `target_dim=15`
- ssmis: `input_dim=33`, `target_dim=24`
- seviri: `input_dim=25`, `target_dim=16`
- avhrr: `input_dim=12`, `target_dim=3`
- ascat: `input_dim=16`, `target_dim=3`
- surface_obs: `input_dim=13`, `target_dim=5`
- radiosonde: `input_dim=12`, `target_dim=4`
- aircraft: `input_dim=12`, `target_dim=4`

Notes:
- Radiosonde/aircraft get an additional learned pressure-level embedding in the model before the input MLP.
- Decoder initialization is geometry-conditioned via scan-angle/edge features.

## 5) Quick "picture" summary (text-only)

```text
[Instrument inputs]
   -> per-instrument Input MLP (to 128)
   -> per-instrument Obs->Mesh GAT encoder
   -> shared mesh latent state
   -> 10x mesh InteractionNetwork layers
   -> sliding-window temporal Transformer (T=4)
   -> per-instrument Mesh->Target GAT decoder
   -> per-instrument Output MLP
   -> instrument-specific targets
```

If you want, we can also generate a Graphviz `.svg` of this exact diagram in-repo for presentations.
