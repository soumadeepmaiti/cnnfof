#Model.py

import torch
import torch.nn as nn

#--
def pos_int_to_base(x, b=2, extend_zero_to_length=0):
    if x == 0:
        return [0]
    out = []
    cur = x
    while cur > 0:
        out.append(int(cur % b))
        cur = int(cur / b)
    if extend_zero_to_length > 0 and len(out) < extend_zero_to_length:
        out += [0] * (extend_zero_to_length - len(out))
    return out[::-1]

#--
def compute_index_from_pad_region(region_code, dim_shape, dim_pad):
    original_lower = 0
    original_upper = 0
    out_lower = 0
    out_upper = 0
    
    if region_code == 1:
        out_upper = dim_pad[0]
        original_lower = dim_shape - dim_pad[0]
        original_upper = dim_shape
        return original_lower, original_upper, out_lower, out_upper

    if region_code == 2:
        out_lower = dim_shape + dim_pad[0]
        out_upper = dim_shape + dim_pad[0] + dim_pad[1]
        original_upper = dim_pad[1]
        return original_lower, original_upper, out_lower, out_upper

    original_upper = dim_shape
    out_lower = dim_pad[0]
    out_upper = dim_shape + dim_pad[0]
    
    return original_lower, original_upper, out_lower, out_upper

#--
def periodic_padding_3d(x, pad):
    ndim = 3
    m = torch.nn.ConstantPad3d(pad, 0)
    out = m(x)
    
    for i in range(1, 3**ndim):
        region_code = pos_int_to_base(i, 3, ndim)
        x_original_lower, x_original_upper, x_out_lower, x_out_upper =\
            compute_index_from_pad_region(region_code[0], int(x.shape[2]), pad[0:2])
        y_original_lower, y_original_upper, y_out_lower, y_out_upper =\
            compute_index_from_pad_region(region_code[1], int(x.shape[3]), pad[2:4])
        z_original_lower, z_original_upper, z_out_lower, z_out_upper =\
            compute_index_from_pad_region(region_code[2], int(x.shape[4]), pad[4:6])
        
        if x_out_lower != x_out_upper and\
            y_out_lower != y_out_upper and\
            z_out_lower != z_out_upper:

            out[:,:,x_out_lower: x_out_upper,
            y_out_lower: y_out_upper,
            z_out_lower: z_out_upper] =\
            x[:,:,x_original_lower: x_original_upper,
                y_original_lower: y_original_upper,
                z_original_lower: z_original_upper]
    
    return out

#--
class BasicBlockHaloClassification(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, conv=True):
        super(BasicBlockHaloClassification, self).__init__()
        self.conv = conv
        self.pad = ((kernel_size - 1) // 2, ) * 6
        self.crop_ref = (kernel_size // 2, kernel_size // 2 + kernel_size % 2)

        if conv:
            self.convol = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.bnD = nn.BatchNorm3d(out_channels)
        else:
            self.deconvol = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.bnU = nn.BatchNorm3d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.conv:
            x = periodic_padding_3d(x, pad=self.pad)
            out = self.convol(x)
            out = self.bnD(out)
        else:
            x = periodic_padding_3d(x, pad=(0,1,0,1,0,1))
            out = self.deconvol(x)
            out = crop_tensor(out, self.crop_ref)
            out = self.bnU(out)
        out = self.relu(out)
        return out


def crop_tensor(x, ref=(1, 2)):
    rstart, rend = ref
    x = x.narrow(2, rstart, x.shape[2] - rstart - rend).narrow(3, rstart, x.shape[3] - rstart - rend).narrow(4, rstart, x.shape[4] - rstart - rend).contiguous()
    return x

#--
class He2019ClassificationNet(nn.Module):
    def __init__(self, block, in_channels=6, out_channels=1, kernel_size=3, outer_stride=1):
        super(He2019ClassificationNet, self).__init__()
        self.layer1a = block(in_channels, 64, kernel_size=kernel_size, stride=outer_stride)
        self.layer1b = block(64, 64, kernel_size=kernel_size)
        self.layer2 = block(64, 128, stride=2, kernel_size=kernel_size)  # downsampling
        self.layer3a = block(128, 128, kernel_size=kernel_size)
        self.layer3b = block(128, 128, kernel_size=kernel_size)
        self.layer4 = block(128, 256, stride=2, kernel_size=kernel_size)  # downsampling
        self.layer5a = block(256, 256, kernel_size=kernel_size)
        self.layer5b = block(256, 256, kernel_size=kernel_size)
        self.layerA = block(256, 128, stride=2, conv=False, kernel_size=kernel_size)  # upsampling
        self.layer6a = block(256, 128, kernel_size=kernel_size)
        self.layer6b = block(128, 128, kernel_size=kernel_size)
        self.layerB = block(128, 64, stride=2, conv=False, kernel_size=kernel_size)  # upsampling
        self.layer7a = block(128, 64, kernel_size=kernel_size)
        self.layer7b = block(64, 64, kernel_size=kernel_size)
        self.layerC = nn.ConvTranspose3d(64, out_channels, kernel_size=outer_stride, stride=outer_stride, padding=0, bias=True)
        # Changed to Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x1 = self.layer1a(x)
        x1 = self.layer1b(x1)
        x = self.layer2(x1)
        x2 = self.layer3a(x)
        x2 = self.layer3b(x2)
        x = self.layer4(x2)
        x = self.layer5a(x)
        x = self.layer5b(x)
        x = self.layerA(x)
        x = torch.cat((x, x2), dim=1)  # skip connection
        x = self.layer6a(x)
        x = self.layer6b(x)
        x = self.layerB(x)
        x = torch.cat((x, x1), dim=1)  # skip connection
        x = self.layer7a(x)
        x = self.layer7b(x)
        x = self.layerC(x)
        x = self.sigmoid(x)  # Apply sigmoid activation for binary classification output
        return x