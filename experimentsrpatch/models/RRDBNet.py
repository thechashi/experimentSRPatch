"""
Residual Residual Dense Generator that takes Noise vector and generates a medical image
"""
import functools
import toml
import torch
import torch.nn as nn

def load(model, path):
    print(path)
    path = 'saved_models/rrdb_gen_500.pt'
    model.load_state_dict(torch.load(path))
    
def make_layer(block, n_layers):
    """

    :param n_layers: No of RRDB Layers
    :return: returns the Sequential model
    """
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """
    Residual Convolution Block
    """

    def __init__(self, nf):
        """

        :param nf: no of input features
        """
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(
            nf + nf, nf, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv3 = nn.Conv2d(
            nf + 2 * nf, nf, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv4 = nn.Conv2d(
            nf + 3 * nf, nf, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv5 = nn.Conv2d(
            nf + 4 * nf, nf, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.p_relu = nn.PReLU()

    def forward(self, input_tensor):
        """

        :param input_tensor: input tensor
        :return:  returns a tensor which is of same shape with added features
        """
        x_1 = self.p_relu(self.conv1(self.pad(input_tensor)))
        x_2 = self.p_relu(self.conv2(self.pad(torch.cat((input_tensor, x_1), 1))))
        x_3 = self.p_relu(self.conv3(self.pad(torch.cat((input_tensor, x_1, x_2), 1))))
        x_4 = self.p_relu(
            self.conv4(self.pad(torch.cat((input_tensor, x_1, x_2, x_3), 1)))
        )
        x_5 = self.p_relu(
            self.conv5(self.pad(torch.cat((input_tensor, x_1, x_2, x_3, x_4), 1)))
        )
        return x_5 * 0.2 + input_tensor


class RRDB(nn.Module):
    """
    Creates the RRDB module from the residual blocks
    """

    def __init__(self, no_fea):
        """

        :param no_fea: No of input features
        """
        super().__init__()
        self.rrdb_1 = ResBlock(no_fea)
        self.rrdb_2 = ResBlock(no_fea)
        self.rrdb_3 = ResBlock(no_fea)

    def forward(self, input_tensor):
        """

        :param  input_tensor: input tensor
        :return:  returns a tensor which is of same shape with
                  added features after Three RRDB passes
        """
        out = self.rrdb_1(input_tensor)
        out = self.rrdb_2(out)
        out = self.rrdb_3(out)
        return out * 0.2 + input_tensor


# def make_layer(block, n_layers):


#    :param n_layers: No of RRDB Layers
#   :return: returns the Sequential model

# layers = []
# for _ in range(n_layers):
#    layers.append(block())
# return nn.Sequential(*layers)


class RRDBNet(nn.Module):
    """
    Residual Network to up-sample the given image
    """

    def __init__(self, in_nc, out_nc, nf, nl):
        """

        :param in_nc: No of input channels of the image
        :param out_nc: No of output channels expected
        :param nf: No of features
        :param nl: No of layers from RRDB
        """
        super().__init__()
        rrdb_block_f = functools.partial(RRDB, no_fea=nf)
        self.conv_first = nn.Conv2d(
            in_nc, nf, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.rrdb_trunk = make_layer(rrdb_block_f, nl)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(
            nf // 4, nf, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.upconv2 = nn.Conv2d(
            nf // 4, nf, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv_hr = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_last = nn.Conv2d(
            nf, out_nc, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.p_relu = nn.PReLU()

    def forward(self, input_tensor):
        """

        :param input_tensor: input Image
        :return: upsampled Image tensor
        """
        fea = self.conv_first(input_tensor)
        trunk = self.trunk_conv(self.rrdb_trunk(fea))
        fea = fea + trunk
        pixel_shuffle = nn.PixelShuffle(2)
        fea = self.p_relu(self.upconv1(pixel_shuffle(fea)))
        fea = self.p_relu(self.upconv2(pixel_shuffle(fea)))
        out = self.conv_last(self.p_relu(self.conv_hr(fea)))

        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except RuntimeError:
                    if name.find("tail") == -1:
                        error_string = (
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}.".format(
                                name, own_state[name].size(), param.size()
                            )
                        )
                        raise error_string from RuntimeError()
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))


class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The channel adapting layer of a Generator -
    maps each the output to the desired number of output channels
    :param input_channels: the number of channels to expect from a given input
    :param output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.conv(self.pad(x))
        return x


class ResidualBlock(nn.Module):
    '''
    :param input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.batchnorm = nn.BatchNorm2d(input_channels)
        self.activation = nn.PReLU()

    def forward(self, input_tensor):
        '''
        :param input_tensor: image tensor of shape (batch size, channels, height, width)
        '''
        original_x = input_tensor.clone()
        input_tensor = self.conv1(input_tensor)
        input_tensor = self.batchnorm(input_tensor)
        input_tensor = self.activation(input_tensor)
        input_tensor = self.conv2(input_tensor)
        input_tensor = self.batchnorm(input_tensor)
        return original_x + input_tensor


class Embedding_Custom(nn.Module):
    """This class is a resblock embedding layer"""
    def __init__(self, in_nc, out_nc, nf, no_of_layers):
        """

        :param in_nc: no. of input channels
        :param out_nc: no. of output channels
        :param nf: number of features
        :param no_of_layers: Number of resblock layers
        """
        super().__init__()
        self.first_conv = FeatureMapBlock(in_nc, nf)
        resblocks = nn.ModuleList()
        for _ in range(no_of_layers):
            resblocks.append(ResidualBlock(nf))
        self.resblocks = nn.Sequential(*resblocks)
        self.last_conv = FeatureMapBlock(nf, out_nc)


class TransferLearning_Generator(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nl, resblock_layers, device, load_model=False, pt_file=None):
        """
        This class will basically take the model and add embedding layers in front and back of model and will train
        these embeddings to transform the image to values that the model can then upsample
        :param in_nc: no. of input channels
        :param out_nc: no. of output channels
        :param nf: no. of features
        :param nl: no. of layers
        :param load_model: set if the middle model needs to be freezed or not
        :param pt_file: give the model pt file
        :param device: The device the model needs to be loaded into
        """
        self.embedder_front = Embedding_Custom(in_nc, in_nc, nf, resblock_layers)
        self.model = RRDBNet(in_nc, out_nc, nf, nl)
        if load_model:
            self.model = self.model.to(device)
            self.model.load_state_dict(pt_file)
            for param in self.model.parameters():
                param.requires_grad = False
            del pt_file
        self.embedder_back = Embedding_Custom(out_nc, out_nc, nf, resblock_layers)

    def forward(self, input_tensor):
        input_tensor = self.embedder_front(input_tensor)
        input_tensor = self.model(input_tensor)
        input_tensor = self.embedder_back(input_tensor)
        return input_tensor