from .lora_loader_node import NODE_CLASS_MAPPINGS as LoraMappings
from .lora_loader_node import NODE_DISPLAY_NAME_MAPPINGS as LoraDisplayMappings

NODE_CLASS_MAPPINGS = {
    **LoraMappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LoraDisplayMappings,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
