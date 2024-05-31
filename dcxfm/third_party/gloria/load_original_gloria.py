import torch
import os
from typing import Union
from . import builder


# FIXME: this path is not correct
_MODELS = {
    "gloria_resnet50": "/home/fywang/Documents/CXRSeg/pretrained/gloria/chexpert_resnet50.ckpt",
}


def load_gloria(
    name: str = "gloria_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available(
    ) else "cpu",
):
    """Load a GLoRIA model

    Parameters
    ----------
    name : str
        A model name listed by `gloria.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    gloria_model : torch.nn.Module
        The GLoRIA model
    """

    # warnings
    if name in _MODELS:
        ckpt_path = _MODELS[name]
    elif os.path.isfile(name):
        ckpt_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"Model {name} not found.\n"
            + "Make sure to download the pretrained weights from \n"
            + "    https://stanfordmedicine.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh \n"
            + " and copy it to the ./pretrained folder."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    gloria_model = builder.build_gloria_model(cfg).to(device)
    gloria_model.load_state_dict(ckpt_dict, strict=False)

    return gloria_model


if __name__ == "__main__":
    model = load_gloria()
    print(model)
