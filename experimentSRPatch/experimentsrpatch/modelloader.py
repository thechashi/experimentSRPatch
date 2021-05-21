import models.EDSR as edsr

def load_edsr(device, n_resblocks=16, n_feats=64):
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
        "n_resblocks": n_resblocks,
        "n_feats": n_feats,
        "scale": [4],
        "rgb_range": 255,
        "n_colors": 3,
        "res_scale": 1,
    }
    model = edsr.make_model(args).to(device)
    edsr.load(model)
    print("\nModel details: ")
    print(model)
    print()
    return model