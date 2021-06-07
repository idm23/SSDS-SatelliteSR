import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.glow.act_norm import ActNorm


class Coupling16(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.

    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate activation
            in NN.
    """
    def __init__(self, in_channels, cond_channels, mid_channels):
        super(Coupling16, self).__init__()
        self.nn = NN(in_channels, cond_channels, mid_channels, 2 * in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, x_cond, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id, x_cond)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj


class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, cond_channels, mid_channels, out_channels,
                 use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d
        
        self.encoder = torchvision.models.vgg16(True).features
        #self.conv1 = nn.Sequential(self.encoder[0], self.relu, self.encoder[2], self.relu) #64
        #self.conv2 = nn.Sequential(self.encoder[5], self.relu, self.encoder[7], self.relu) #128
        
        weight1 = self.encoder[0].weight
        dif_size = in_channels - 3
        while dif_size/3 > 1:
            weight1 = torch.cat((weight1, self.encoder[0].weight), dim = 1)
            dif_size -= 3
        if dif_size > 0:
            weight1 = torch.cat((weight1, self.encoder[0].weight[:,0:dif_size,:,:]), dim = 1)
            
        weight2 = self.encoder[0].weight
        dif_size = cond_channels - 3
        while dif_size/3 > 1:
            weight2 = torch.cat((weight2, self.encoder[0].weight), dim = 1)
            dif_size -= 3
        if dif_size > 0:
            weight2 = torch.cat((weight2, self.encoder[0].weight[:,0:dif_size,:,:]), dim = 1)
        
        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1, bias = False)
            )
        self.in_condconv = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1, bias = False)
            )
        self.in_conv[0].weight = torch.nn.Parameter(weight1)
        self.in_conv[1].weight = torch.nn.Parameter(self.encoder[2].weight)
        self.in_conv[2].weight = torch.nn.Parameter(self.encoder[5].weight)
        self.in_condconv[0].weight = torch.nn.Parameter(weight2)
        self.in_condconv[1].weight = torch.nn.Parameter(self.encoder[2].weight)
        self.in_condconv[2].weight = torch.nn.Parameter(self.encoder[5].weight)
        
        self.mid_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.mid_condconv1 = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1, bias = False)
            )
        self.mid_conv1.weight = torch.nn.Parameter(self.encoder[7].weight)
        self.mid_condconv1[0].weight = torch.nn.Parameter(weight2)
        self.mid_condconv1[1].weight = torch.nn.Parameter(self.encoder[2].weight)
        self.mid_condconv1[2].weight = torch.nn.Parameter(self.encoder[5].weight)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.mid_condconv2 = nn.Conv2d(cond_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv2.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv2.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, x_cond):
        
        x = self.in_norm(x)
        x = self.in_conv(x) + self.in_condconv(x_cond)
        x = F.relu(x)

        x = self.mid_conv1(x) + self.mid_condconv1(x_cond)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.mid_conv2(x) + self.mid_condconv2(x_cond)
        x = self.out_norm(x)
        x = F.relu(x)

        x = self.out_conv(x)

        return x
