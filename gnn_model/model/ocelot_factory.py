
import yaml

from gnn_model.model.mesh import mesh
from ..configs.model_config import ModelConfig, MeshConfig, CoderConfig, ProcessorConfig, EmbeddingsConfig
from model.mesh.fixed_mesh import FixedMesh
from model.mesh.hierarchical_mesh import HierarchicalMesh
from model.coder import GATCoder, InteractionCoder
from model.processor import SlidingTransformerProcessor, InteractionProcessor, HierarchicalProcessor, HierarchicalTransformerProcessor
from gnn_model.model.ocelot import Ocelot


class OcelotFactory:
    @staticmethod
    def create_model(config_path: str, verbose: bool = False) -> Ocelot:
        model_config = ModelConfig(config_path)
        
        encoder = OcelotFactory._create_coder(model_config.encoder)
        mesh = OcelotFactory._create_mesh(model_config.mesh)
        processor = OcelotFactory._create_processor(model_config.processor, mesh)
        decoder = OcelotFactory._create_coder(model_config.decoder)

        return Ocelot(encoder=encoder, 
                      mesh=mesh, 
                      processor=processor, 
                      decoder=decoder,
                      verbose=verbose)

    @staticmethod
    def _create_coder(coder_config: CoderConfig) -> Coder:
        if coder_config.type == 'gat':
            return GATCoder(layers=coder_config.layers, heads=coder_config.heads, dropout=coder_config.dropout)
        elif coder_config.type == 'interaction':
            return InteractionCoder(layers=coder_config.layers, heads=coder_config.heads, dropout=coder_config.dropout)
        else:
            raise ValueError(f"Unknown coder type: {coder_config.type}")
        
    @staticmethod
    def _create_mesh(mesh_config: MeshConfig) -> mesh:
        if mesh_config.type == 'fixed':
            return FixedMesh(levels=mesh_config.levels, splits=mesh_config.resolution)
        elif mesh_config.type == 'hierarchical':
            return HierarchicalMesh(levels=mesh_config.levels, splits=mesh_config.resolution)
        else:
            raise ValueError(f"Unknown mesh type: {mesh_config.type}")
        
    @staticmethod
    def _create_processor(processor_config: ProcessorConfig, mesh: mesh.Mesh) -> Processor:
        if processor_config.type == 'sliding_transformer' or processor_config.type == 'interaction':
            if not isinstance(mesh, FixedMesh):
                raise ValueError(f"Processor type '{processor_config.type}' requires a fixed mesh")
        elif processor_config.type == 'hierarchical' or processor_config.type == 'hierarchical_transformer':
            if not isinstance(mesh, HierarchicalMesh):
                raise ValueError(f"Processor type '{processor_config.type}' requires a hierarchical mesh")
        
        if processor_config.type == 'sliding_transformer':
            return SlidingTransformerProcessor(num_layers=processor_config.num_layers, depth=processor_config.depth, heads=processor_config.heads, window=processor_config.window, dropout=processor_config.dropout)
        elif processor_config.type == 'interaction':
            return InteractionProcessor(num_layers=processor_config.num_layers, depth=processor_config.depth, heads=processor_config.heads, window=processor_config.window, dropout=processor_config.dropout)
        elif processor_config.type == 'hierarchical':
            return HierarchicalProcessor(num_layers=processor_config.num_layers, depth=processor_config.depth, heads=processor_config.heads, window=processor_config.window, dropout=processor_config.dropout)
        elif processor_config.type == 'hierarchical_transformer':
            return HierarchicalTransformerProcessor(num_layers=processor_config.num_layers, depth=processor_config.depth, heads=processor_config.heads, window=processor_config.window, dropout=processor_config.dropout)
        else:
            raise ValueError(f"Unknown processor type: {processor_config.type}")
        