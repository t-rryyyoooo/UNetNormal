import torch
from torch import nn
from .utils import cropping3D
from torchsummary import summary

class DoubleConvolution(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, n=2, use_bn=True):
        super(DoubleConvolution, self).__init__()

        self.layers = []
        for i in range(1, n + 1):
            if i == 1:
                x = nn.Conv3d(in_channel, mid_channel, (3, 3, 3))
            else:
                x = nn.Conv3d(mid_channel, out_channel, (3, 3, 3))

            self.layers.append(x)

            if use_bn:
                if i == 1:
                    self.layers.append(nn.BatchNorm3d(mid_channel))
                else:
                    self.layers.append(nn.BatchNorm3d(out_channel))
                
            
            self.layers.append(nn.ReLU())

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class CreateConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, n=2, use_bn=True, apply_pooling=True):
        super(CreateConvBlock, self).__init__()
        self.apply_pooling = apply_pooling

        self.DoubleConvolution = DoubleConvolution(in_channel, mid_channel, out_channel, n=2, use_bn=use_bn)

        if apply_pooling:
            self.maxpool = nn.MaxPool3d((2, 2, 2))
        
    def forward(self, x):
        x = self.DoubleConvolution(x)
        conv_result = x
        if self.apply_pooling:
            x = self.maxpool(x)

        return x, conv_result


class CreateUpConvBlock(nn.Module):
    def __init__(self, in_channel, concat_channel, mid_channel, out_channel,  n=2, use_bn=True):
        super(CreateUpConvBlock, self).__init__()

        x = nn.ConvTranspose3d(in_channel, in_channel, (2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), dilation=1)
        self.convTranspose = x

        self.DoubleConvolution = DoubleConvolution(in_channel + concat_channel, mid_channel, out_channel, n=2, use_bn=use_bn)

    def forward(self, x1, x2):
        x1 = self.convTranspose(x1)
        c = [(i - j) for (i, j) in zip(x2.size()[2:], x1.size()[2:])]
        hc = [ i // 2 for i in c]

        x2 = cropping3D(x2, (hc[0], hc[0] if c[0]%2 == 0 else hc[0] + 1), (hc[1], hc[1] if c[1]%2 == 0 else hc[1] + 1), (hc[2], hc[2] if c[2]%2 == 0 else hc[2] + 1))
        x = torch.cat([x2, x1], dim=1)

        x = self.DoubleConvolution(x)

        return x

class UNetModel(nn.Module):
    def __init__(self, in_channel, nclasses, dropout=0.5, use_bn=True, use_dropout=True):
        super(UNetModel, self).__init__()
        self.use_dropout = use_dropout

        self.contracts = []
        self.expands = []

        contract = CreateConvBlock(in_channel, 32, 64, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        contract = CreateConvBlock(64, 64, 128, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        contract = CreateConvBlock(128, 128, 256, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        self.lastContract = CreateConvBlock(256, 256, 512, n=2, use_bn=use_bn, apply_pooling=False)

        self.contracts = nn.ModuleList(self.contracts)

        if use_dropout:
            self.dropout = nn.Dropout(dropout)

        expand = CreateUpConvBlock(512, 256, 256, 256, n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(256, 128, 128, 128, n=2, use_bn=use_bn)
        self.expands.append(expand)
         
        expand = CreateUpConvBlock(128, 64, 64, 64, n=2, use_bn=use_bn)
        self.expands.append(expand)

        self.expands = nn.ModuleList(self.expands)

        self.segmentation = nn.Conv3d(64, nclasses, (1, 1, 1), stride=1, dilation=1, padding=(0, 0, 0))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv_results = []
        for contract in self.contracts:
            x, conv_result = contract(x)
            conv_results.append(conv_result)

        conv_results = conv_results[::-1]
        #conv_results = nn.ModuleList(conv_results)

        x, _ = self.lastContract(x)
        if self.use_dropout:
            x = self.dropout(x)
            
        for expand, conv_result in zip(self.expands, conv_results):
            x = expand(x, conv_result)

        x = self.segmentation(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    model=UNetModel(1 ,3)
    net_shape = (1, 1, 44+ 44*2, 44 + 44*2, 28 + 44*2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    dummy_img = torch.rand(net_shape).to(device)
    print("input: ", net_shape)

    output = model(dummy_img)
    summary(model, net_shape)
    print('output:', output.size())
