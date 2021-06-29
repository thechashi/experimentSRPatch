"""Model Loader"""
import models.EDSR as edsr
import models.RRDBNet as rrdb
import toml
import torch


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
    cpu = "False" if torch.cuda.is_available() else "True"
    # =============================================================================
    #     if cpu == 'True':
    #         print('\nCuda not available\n')
    #     elif cpu == 'False':
    #         print('\nCUDA Available\n')
    # =============================================================================
    args = {
        "n_resblocks": n_resblocks,
        "n_feats": n_feats,
        "scale": [scale],
        "rgb_range": 255,
        "n_colors": 3,
        "res_scale": 1,
        "cpu": cpu,
    }
    model = edsr.make_model(args).to(device)
    edsr.load(model)
    if model_details:
        # =============================================================================
        #         print("\nModel details: ")
        #         print(model)
        # =============================================================================
        print()
    return model


def load_rrdb(device):
    config = toml.load("../gantrainingconfig.toml")
    path_config = toml.load("../config.toml")
    path = path_config["rrdb_path"]
    model = rrdb.RRDBNet(
        config["generator"]["num_in_channels"],
        config["generator"]["num_out_channels"],
        config["generator"]["num_fea"],
        config["generator"]["num_layers"])
    rrdb.load(model, path).to(device)
    return model