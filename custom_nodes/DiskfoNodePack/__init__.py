from .lora_loader_node import NODE_CLASS_MAPPINGS as LoraMappings
from .lora_loader_node import NODE_DISPLAY_NAME_MAPPINGS as LoraDisplayMappings

from .timer_nodes import NODE_CLASS_MAPPINGS as TimerMappings
from .timer_nodes import NODE_DISPLAY_NAME_MAPPINGS as TimerDisplayMappings

NODE_CLASS_MAPPINGS = {
    **LoraMappings,
    **TimerMappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LoraDisplayMappings,
    **TimerDisplayMappings,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
