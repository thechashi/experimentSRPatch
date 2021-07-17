"""Model Loader"""
import models.EDSR as edsr
import models.RRDBNet as rrdb
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
    args = {
        "G0": 64,
        "RDNconfig": "B",
        "RDNkSize": 3,
        "act": "relu",
        "batch_size": 16,
        "betas": (0.9, 0.999),
        "chop": True,
        "cpu": True,
        "data_range": "1-800/801-810",
        "data_test": ["Demo"],
        "data_train": ["DIV2K"],
        "debug": False,
        "decay": "200",
        "dilation": False,
        "dir_data": "../../../dataset",
        "dir_demo": "../test",
        "epochs": 300,
        "epsilon": 1e-08,
        "ext": "sep",
        "extend": ".",
        "gamma": 0.5,
        "gan_k": 1,
        "gclip": 0,
        "load": "",
        "loss": "1*L1",
        "lr": 0.0001,
        "model": "EDSR",
        "momentum": 0.9,
        "n_GPUs": 1,
        "n_colors": 3,
        "n_feats": 64,
        "n_resblocks": 16,
        "n_resgroups": 10,
        "n_threads": 6,
        "no_augment": False,
        "optimizer": "ADAM",
        "patch_size": 192,
        "pre_train": "download",
        "precision": "single",
        "print_every": 100,
        "reduction": 16,
        "res_scale": 1,
        "reset": False,
        "resume": 0,
        "rgb_range": 255,
        "save": "test",
        "save_gt": False,
        "save_models": False,
        "save_results": True,
        "scale": [4],
        "seed": 1,
        "self_ensemble": False,
        "shift_mean": True,
        "skip_threshold": 100000000.0,
        "split_batch": 1,
        "template": ".",
        "test_every": 1000,
        "test_only": True,
        "weight_decay": 0,
    }
    model = edsr.make_model(args).to(device)
    edsr.load(model)
    if model_details:
        pass
    return model


def load_rrdb(device):
    config = toml.load("../gantrainingconfig.toml")
    path_config = toml.load("../config.toml")
    path = path_config["rrdb_path"]
    model = rrdb.RRDBNet(
        config["generator"]["num_in_channels"],
        config["generator"]["num_out_channels"],
        config["generator"]["num_fea"],
        config["generator"]["num_layers"],
    )
    model = model.to(device)
    rrdb.load(model, path)
    return model
