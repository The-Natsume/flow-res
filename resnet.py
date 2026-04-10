import torch
import torch.nn as nn
import torch.nn.functional as F


def from_torch_real_to_torch_complex(x):
    x = x[:,0,:,:]+1j*x[:,1,:,:]
    x = torch.unsqueeze(x,dim=1).to(torch.complex64)
    return x

def from_torch_complex_to_torch_real(x):
    real = torch.real(x)
    imag = torch.imag(x)
    data = torch.cat((real, imag), dim=1).to(torch.float32)
    return data


def DC(x, data, mask):
    x = torch.fft.ifft2(torch.fft.fft2(x) * (1 - mask) + torch.fft.fft2(data) * mask)
    return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, apply_relu=False, apply_scaling=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        self.conv_weight = nn.Conv2d(in_channels, out_channels*3, kernel_size, stride=1, padding=1)

        self.apply_relu = apply_relu
        self.apply_scaling = apply_scaling
        self.scaling_factor = 0.1

    def forward(self, x):

        if self.apply_scaling:

            gamma1, scale1, shift1 = torch.chunk(self.conv_weight(x), 3, dim=1)
            x = (self.conv(x.mul(scale1.add(1))).add_(shift1)).mul_(gamma1)
            # x = (x.mul(scale1.add(1)).add_(shift1)).mul_(gamma1)
            if self.apply_relu:
                x = F.relu(x)
        else:
            x = self.conv(x)

            if self.apply_relu:
                x = F.relu(x)

        return x

class MuParam(nn.Module):
    def __init__(self):
        super(MuParam, self).__init__()
        self.mu = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self):
        return self.mu

class ResNet(nn.Module):
    def __init__(self, nb_res_blocks=15,input_dim=2,inner_dim =64):
        super(ResNet, self).__init__()
        self.nb_res_blocks = nb_res_blocks

        # First Layer
        self.first_layer = ConvLayer(input_dim, inner_dim, 3, apply_relu=False, apply_scaling=False)

        # Residual Blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ConvLayer(inner_dim, inner_dim, 3, apply_relu=True, apply_scaling=False),
                ConvLayer(inner_dim, inner_dim, 3, apply_relu=False, apply_scaling=True)
            ) for _ in range(nb_res_blocks)
        ])

        # Last Layer
        self.last_layer = ConvLayer(inner_dim, inner_dim, 3, apply_relu=False, apply_scaling=False)

        # Residual
        self.residual_layer = ConvLayer(inner_dim, 2, 3, apply_relu=False, apply_scaling=False)

    def forward(self, x):
        # First Layer
        x0 = self.first_layer(x)

        # Residual Blocks
        x_prev = x0
        for block in self.res_blocks:
            conv1 = block[0](x_prev)
            conv2 = block[1](conv1)
            x_prev = conv2 + x_prev

        # Last Layer
        rb_output = self.last_layer(x_prev)

        # Residual
        temp_output = rb_output + x0
        nw_output = self.residual_layer(temp_output)

        return nw_output


class DnCn_resnet(nn.Module):
    def __init__(self, input_dim=2, num=5,inner_dim=64,dc=False,**kwargs):
        super(DnCn_resnet, self).__init__()
        self.nc = num
        conv_blocks = []
        dcs = []
        self.dc = dc
        for i in range(num):
            conv_blocks.append(ResNet(input_dim=input_dim,inner_dim=inner_dim))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,cond=None,m=None):
        x_in = x
        if cond is not None:
            x_cat = torch.cat([x_in,cond],dim=1)
            q = 0
            for i in range(self.nc):
                x_cnn = self.conv_blocks[i](x_cat)
                x_in = x_in + x_cnn

                if self.dc and m is not None:

                    x_in = from_torch_complex_to_torch_real(DC(from_torch_real_to_torch_complex(x_in),from_torch_real_to_torch_complex(x),m))
                x_cat = torch.cat([x_in,cond],dim=1)
        else:

            for i in range(self.nc):
                x_cnn = self.conv_blocks[i](x_in)
                x_in = x_in + x_cnn

                if self.dc and m is not None:
                    x_in = from_torch_complex_to_torch_real(
                        DC(from_torch_real_to_torch_complex(x_in), from_torch_real_to_torch_complex(x), m))
        return x_in