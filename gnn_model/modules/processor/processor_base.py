
import torch.nn as nn
from torch_geometric.data import HeteroData

from logger import log


class ProcessorBase(nn.Module):
    def __init__(self, mesh, enable_mesh_pred: bool = False):
        super().__init__()
        self.mesh = mesh
        self.enable_mesh_pred = enable_mesh_pred

    @staticmethod
    def _get_latent_step_info(data: HeteroData) -> dict:
        """
        Extract information about latent steps from the batch.
        Returns dict with step mapping and number of steps.
        """
        step_info = {}
        max_step = -1

        # Find all step-specific target nodes and map them to base instruments
        for node_type in data.node_types:
            if "_target_step" in node_type:
                # Extract: atms_target_step0 -> (atms_target, 0)
                parts = node_type.split("_step")
                if len(parts) == 2:
                    base_type = parts[0]  # e.g., "atms_target"
                    try:
                        step_num = int(parts[1])
                        if base_type not in step_info:
                            step_info[base_type] = {}
                        step_info[base_type][step_num] = node_type
                        max_step = max(max_step, step_num)
                    except ValueError:
                        continue

        return {
            "step_mapping": step_info,
            "num_steps": max_step + 1 if max_step >= 0 else 0
        }

    def _generate_predictions(self, data: HeteroData, step_mapping: dict, mesh_features_processed) -> dict:
        # Initialize predictions dict with lists for each base instrument
        predictions = {}
        for base_type in step_mapping.keys():
            predictions[base_type] = []


        # Process all instruments for this step
        for base_type, steps_dict in step_mapping.items():
            if step in steps_dict:
                step_node_type = steps_dict[step]  # e.g., "atms_target_step0"

                # Find the corresponding edge
                step_edge_type = None
                step_edge_index = None
                for edge_type, edge_index in data.edge_index_dict.items():
                    src_type, _, dst_type = edge_type
                    if src_type == "mesh" and dst_type == step_node_type:
                        step_edge_type = edge_type
                        step_edge_index = edge_index
                        print(f"decode: [edge_type] {edge_type}: {edge_index.shape}")
                        break

                if step_edge_type is None or step_edge_index is None:
                    log.debug(f"[LATENT] Warning: No edge found for {step_node_type}")
                    continue

                # Get the decoder (mapped to base instrument)
                decoder_key = edge_mapping.get(step_edge_type)
                if decoder_key not in self.observation_decoders:
                    log.debug(f"[LATENT] Warning: No decoder found for {decoder_key}")
                    continue

                decoder = self.observation_decoders[decoder_key]
                decoder.edge_index = step_edge_index

                # Condition decoder on viewing geometry at initialization
                # - For satellites: viewing zenith angle (scan angle)
                # - For radiosonde/aircraft: pressure level (vertical viewing geometry)
                reference_device = mesh_features_processed.device
                N = data[step_node_type].num_nodes

                # Embed viewing geometry information FIRST (before decoder initialization)
                sa_emb = None
                pressure_emb = None
                time_emb = None
                if base_type == "ascat_target":
                    scan_angle = data[step_node_type].x  # [N,3] for ASCAT
                    sa_emb = self.ascat_scan_angle_embedder(scan_angle)  # [N, scan_embed_dim]
                elif base_type in ("atms_target", "amsua_target", "avhrr_target", "cris_pca_target", "seviri_asr_target", "seviri_csr_target"):
                    scan_angle = data[step_node_type].x  # [N,1] for ATMS/AMSU-A/AVHRR/CrIS-PCA
                    sa_emb = self.scan_angle_embedder(scan_angle)  # [N, scan_embed_dim]

                    # Diagnostic: verify scan angle varies
                    if base_type == "atms_target" and self.global_step % 200 == 0:
                        sa = data[step_node_type].x
                        if sa.numel() == 0:
                            print(f"[SCAN DIAG] scan_angle: shape={sa.shape} (empty)")
                        else:
                            sa_f = sa.float()
                            mean_v = sa_f.mean().item()
                            std_v = sa_f.std(unbiased=False).item()
                            min_v = sa_f.min().item()
                            max_v = sa_f.max().item()
                            print(
                                f"[SCAN DIAG] scan_angle: shape={sa.shape}, mean={mean_v:.4f}, "
                                f"std={std_v:.4f}, min={min_v:.4f}, max={max_v:.4f}"
                            )
                elif base_type in ["radiosonde_target", "aircraft_target"] and "pressure_level" in data[step_node_type]:
                    # For radiosonde and aircraft: condition on pressure level (vertical geometry)
                    pressure_level_idx = data[step_node_type].pressure_level  # [N]
                    pressure_emb = self.pressure_level_embedder(pressure_level_idx)  # [N, pressure_embed_dim=8]

                if hasattr(data[step_node_type], "target_metadata"):
                    target_metadata = data[step_node_type].target_metadata
                    if (
                        target_metadata is not None
                        and target_metadata.numel() > 0
                        and target_metadata.size(1) >= (2 + self.target_time_feature_dim)
                    ):
                        time_feat = target_metadata[:, -self.target_time_feature_dim:].to(reference_device)
                        time_emb = self.target_time_embedder(time_feat)

                # Decoder initialization: CONDITION on viewing geometry
                # Instead of zeros, initialize decoder WITH geometry information
                if sa_emb is not None:
                    # Satellite: condition decoder on scan angle (viewing zenith angle)
                    if self.scan_angle_projector is not None:
                        target_features_initial = self.scan_angle_projector(sa_emb)
                    else:
                        # Backward-compatible behavior: scan info only in the last dims.
                        padding_dim = self.hidden_dim - self.scan_angle_embed_dim
                        target_features_initial = torch.cat([
                            torch.zeros(N, padding_dim, device=reference_device),
                            sa_emb
                        ], dim=-1)  # [N, hidden_dim] with scan info in last 8 dims
                elif pressure_emb is not None:
                    # Radiosonde/Aircraft: condition decoder on pressure level (vertical viewing geometry)
                    # Make prediction explicitly depend on geometry
                    if self.pressure_level_projector is not None:
                        target_features_initial = self.pressure_level_projector(pressure_emb)
                    else:
                        padding_dim = self.hidden_dim - self.pressure_level_embed_dim
                        target_features_initial = torch.cat([
                            torch.zeros(N, padding_dim, device=reference_device),
                            pressure_emb
                        ], dim=-1)  # [N, hidden_dim] with pressure info in last 8 dims
                else:
                    # Conventional obs without viewing geometry: use zeros
                    target_features_initial = torch.zeros(N, self.hidden_dim, device=reference_device)

                # Add target-time conditioning as an additive bias over the full hidden_dim.
                if time_emb is not None:
                    target_features_initial = target_features_initial + self.target_time_projector(time_emb)

                edge_attr = self._edge_features(
                    data=data,
                    edge_type=step_edge_type,
                    edge_index=step_edge_index,
                    device=reference_device,
                    dtype=mesh_features_processed.dtype,
                )

                # Decoder now receives GEOMETRY-CONDITIONED initialization
                # This ensures the model CANNOT make predictions without knowing viewing geometry
                decoded_target_features = decoder(
                    send_rep=mesh_features_processed,
                    rec_rep=target_features_initial,  # NOW conditioned on viewing geometry!
                    edge_rep=edge_attr,
                )

                # Decoder output goes directly to output mapper
                # The model learns to use the geometry information that's embedded in target_features_initial

                # Diagnostic logging for radiosonde
                if base_type == "radiosonde_target" and pressure_emb is not None and self.global_step % 200 == 0:
                    print(f"[GRAPHDOP] Radiosonde: decoder conditioned on pressure (decoded shape={decoded_target_features.shape})")

                # Diagnostic logging for satellites
                if base_type == "atms_target" and sa_emb is not None and self.global_step % 200 == 0:
                    print(f"ATMS: decoder conditioned on scan angle (decoded shape={decoded_target_features.shape})")

                # Safety: verify mapper exists before using
                assert base_type in self.output_mappers, f"Missing output mapper for {base_type}"
                step_prediction = self.output_mappers[base_type](decoded_target_features)

                # Store prediction for this step
                predictions[base_type].append(step_prediction)
                print(f"predict: [node_type] {base_type}: {step_prediction.shape}")

                log.debug(f"[LATENT] Step {step} - {base_type}: {step_prediction.shape}")

        return predictions

