import click
from pathlib import Path
import torch
import downloader as dw
import toml

def load_model(model_name, model_pt, image_path, input_channel, output_channel, num_fea, num_layers, device="cuda"):
    model_module = "models." + model_name + ".RRDBNet"
# =============================================================================
#     cus_model = __import__(model_module)
#     
#     model_class = getattr(cus_model, 'RRDBNet')
#     model_class = getattr(cus_model, 'RRDBNet')
# =============================================================================
    model_class = get_class(model_module)    
    model = model_class(input_channel, output_channel, num_fea, num_layers)
    model = model.to(device)
    load_pt_file(model, model_pt)
    return model

def load_pt_file(model, pt_path):
    model_path = Path(pt_path)
    if model_path.exists():
        checkpoint = torch.load(pt_path)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
    else:
        downloader_dict = toml.load('../downloader.toml')
        dw.data_download(downloader_dict['datadict'], './downloads')
        checkpoint = torch.load(pt_path)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m
        
@click.command()
@click.option("--model_name", default="RRDBNet", help="Name of the model")
@click.option("--model_pt", default="saved_models/rrdb_gen_500.pt", help="Path of the model's pt file")
@click.option("--image_path", default="data/slices/0.npz", help="Image path")
@click.option("--input_channel", default=1, help="Number of input channel")
@click.option("--output_channel", default=1, help="Number of output channel")
@click.option("--num_fea", default=64, help="Number of features")
@click.option("--num_layers", default=23, help="Number of layers")
def main(model_name, model_pt, image_path, input_channel, output_channel, num_fea, num_layers, device="cuda"):
    model = load_model(model_name, model_pt, image_path, input_channel, output_channel, num_fea, num_layers, device="cuda")
    print(model)
if __name__ == "__main__":
    main()
    

