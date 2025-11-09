"""
The `stamp.modeling` module implements classes for modeling approaches.
"""

# import all available model approach classes
from .stamp_trainer import STAMPModelingApproach
from .cbramod import CBraModModelingApproach
from .moment_lora_adapter import MOMENTLoraAdapterModelingApproach


# Create a dictionary that maps modeling approach names to their corresponding class objects
modeling_approach_classes = {
    "STAMPModelingApproach": STAMPModelingApproach,
    "CBraModModelingApproach": CBraModModelingApproach,
    "MOMENTLoraAdapterModelingApproach": MOMENTLoraAdapterModelingApproach
}

def create_modeling_approach(
    modeling_approach_config:dict,
    )->any:
    """Retrieve a modeling approach object based on the given name.

    Args:
        modeling_approach_config (dict):

    Raises:
        ValueError: If the given name is not available in modeling_approach_classes.

    Returns:
        any: A modeling approach class that contains train() and predict() functions.
    """
    # Get the class object from the dictionary using the input string
    modeling_approach_class = modeling_approach_classes.get(modeling_approach_config['modeling_approach_name'])

    if modeling_approach_class is None:
        raise ValueError(f"Unknown modeling approach: {modeling_approach_config['modeling_approach_name']}")

    return modeling_approach_class(**modeling_approach_config['params'])

__all__ = [
    "create_modeling_approach"
]
