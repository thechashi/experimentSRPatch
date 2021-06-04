"""Model Loader"""
import models.EDSR as edsr
import toml

def load_edsr(device, n_resblocks=16, n_feats=64, model_details=True):
    """
    Loads the EDSR model

    Parameters
    ----------
    device : str
        device type.
    n_resblocks : int, optional
        number of res_blocks. The default is 16.
    n_feats : int, optional
        number of features. The default is 64.

    Returns
    -------
    model : torch.nn.model
        EDSR model.

    """
    config = toml.load("../config.toml")
    scale = int(config["scale"]) if config["scale"] else 4
    args = {
        "n_resblocks": n_resblocks,
        "n_feats": n_feats,
        "scale": [scale],
        "rgb_range": 255,
        "n_colors": 3,
        "res_scale": 1,
    }
    model = edsr.make_model(args).to(device)
    edsr.load(model)
    if model_details:
        print("\nModel details: ")
# =============================================================================
#         print(model)
# =============================================================================
        print()
    return model
