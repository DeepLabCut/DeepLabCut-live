"""
Helper function to deal with default values in the model configuration.
For instance, "num_bodyparts x 2" is replaced with the number of bodyparts multiplied by 2.
"""

# NOTE - DUPLICATED @deruyter92 2026-01-23: Copied from the original DeepLabCut codebase
# from deeplabcut/pose_estimation_pytorch/modelzoo/utils.py
import copy


def update_config(config: dict, max_individuals: int, device: str):
    """Loads the model configuration file for a model, detector and SuperAnimal

    Args:
        config: The default model configuration file.
        max_individuals: The maximum number of detections to make in an image
        device: The device to use to train/run inference on the model

    Returns:
        The model configuration for a SuperAnimal-pretrained model.
    """
    config = replace_default_values(
        config,
        num_bodyparts=len(config["metadata"]["bodyparts"]),
        num_individuals=max_individuals,
        backbone_output_channels=config["model"]["backbone_output_channels"],
    )
    config["metadata"]["individuals"] = [f"animal{i}" for i in range(max_individuals)]

    config["device"] = device
    if config.get("detector", None) is not None:
        config["detector"]["device"] = device

    return config


def replace_default_values(
    config: dict | list,
    num_bodyparts: int | None = None,
    num_individuals: int | None = None,
    backbone_output_channels: int | None = None,
    **kwargs,
) -> dict:
    """Replaces placeholder values in a model configuration with their actual values.

    This method allows to create template PyTorch configurations for models with values
    such as "num_bodyparts", which are replaced with the number of bodyparts for a
    project when making its Pytorch configuration.

    This code can also do some basic arithmetic. You can write "num_bodyparts x 2" (or
    any factor other than 2) for location refinement channels, and the number of
    channels will be twice the number of bodyparts. You can write
    "backbone_output_channels // 2" for the number of channels in a layer, and it will
    be half the number of channels output by the backbone. You can write
    "num_bodyparts + 1" (such as for DEKR heatmaps, where a "center" bodypart is added).

    The three base placeholder values that can be computed are "num_bodyparts",
    "num_individuals" and "backbone_output_channels". You can add more through the
    keyword arguments (such as "paf_graph": list[tuple[int, int]] or
    "paf_edges_to_keep": list[int] for DLCRNet models).

    Args:
        config: the configuration in which to replace default values
        num_bodyparts: the number of bodyparts
        num_individuals: the number of individuals
        backbone_output_channels: the number of backbone output channels
        kwargs: other placeholder values to fill in

    Returns:
        the configuration with placeholder values replaced

    Raises:
        ValueError: if there is a placeholder value who's "updated" value was not
            given to the method
    """

    def get_updated_value(variable: str) -> int | list[int]:
        var_parts = variable.strip().split(" ")
        var_name = var_parts[0]
        if updated_values[var_name] is None:
            raise ValueError(
                f"Found {variable} in the configuration file, but there is no default value for this variable."
            )

        if len(var_parts) == 1:
            return updated_values[var_name]
        elif len(var_parts) == 3:
            operator, factor = var_parts[1], var_parts[2]
            if not factor.isdigit():
                raise ValueError(f"F must be an integer in variable: {variable}")

            factor = int(factor)
            if operator == "+":
                return updated_values[var_name] + factor
            elif operator == "x":
                return updated_values[var_name] * factor
            elif operator == "//":
                return updated_values[var_name] // factor
            else:
                raise ValueError(f"Unknown operator for variable: {variable}")

        raise ValueError(
            f"Found {variable} in the configuration file, but cannot parse it."
        )

    updated_values = {
        "num_bodyparts": num_bodyparts,
        "num_individuals": num_individuals,
        "backbone_output_channels": backbone_output_channels,
        **kwargs,
    }

    config = copy.deepcopy(config)
    if isinstance(config, dict):
        keys_to_update = list(config.keys())
    elif isinstance(config, list):
        keys_to_update = range(len(config))
    else:
        raise ValueError(f"Config to update must be dict or list, found {type(config)}")

    for k in keys_to_update:
        if isinstance(config[k], (list, dict)):
            config[k] = replace_default_values(
                config[k],
                num_bodyparts,
                num_individuals,
                backbone_output_channels,
                **kwargs,
            )
        elif (
            isinstance(config[k], str)
            and config[k].strip().split(" ")[0] in updated_values.keys()
        ):
            config[k] = get_updated_value(config[k])

    return config
