from ocelot.configs.model_config import CoderConfig
from ocelot.model.coder.attn_bipartite import BipartiteGAT
from ocelot.model.coder.interaction_net import InteractionNet

coder_types = {
    "gat": BipartiteGAT,
    "interaction": InteractionNet
}

def make(config: CoderConfig):
    if config.type not in coder_types:
        raise ValueError(f"Unknown coder type: {config.type}")
        
    print (f"Created {config.type}.")
    return coder_types[config.type](config)
