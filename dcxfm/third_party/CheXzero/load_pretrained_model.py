import torch
import sys
from dcxfm.third_party.CheXzero.model import CLIP
from dcxfm.third_party.CheXzero import clip


def load_clip(model_path, pretrained=False, context_length=77):
    """
    FUNCTION: load_clip
    ---------------------------------
    """
    device = torch.device("cpu")
    if pretrained is False:
        # use new model params
        params = {
            'embed_dim': 768,
            'image_resolution': 320,
            'vision_layers': 12,
            'vision_width': 768,
            'vision_patch_size': 16,
            'context_length': context_length,
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12
        }

        model = CLIP(**params)
    else:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        print("Argument error. Set pretrained = True.", sys.exc_info()[0])
        raise
    return model


def load_pretrained_chexzero(model_path, pretrained=True, context_length=77):
    model = load_clip(
        model_path=model_path,
        pretrained=pretrained,
        context_length=context_length
    )
    return model


if __name__ == "__main__":
    model_path = "/home/fywang/Documents/CXRSeg/pretrained/CheXzero/best_64_0.0001_original_16000_0.861.pt"
    model = load_pretrained_chexzero(
        model_path, pretrained=True, context_length=77)
    print(model)
