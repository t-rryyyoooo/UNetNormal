import torch
from torch import nn
from .model_part import *

class BuildThinUNet(nn.Module):
    def __init__(self, in_channel, out_channel=64, use_bn=True, use_dropout=True):
        super(BuildThinUNet, self).__init__()
        o = out_channel
        self.use_dropout = use_dropout

        self.contracts = []
        self.expands = []

        contract = CreateConvBlock(in_channel, o // 2, o, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        contract = CreateConvBlock(o, o, o*2, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        contract = CreateConvBlock(o*2, o*2, o*4, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        self.lastContract = CreateConvBlock(o*4, o*4, o*8, n=2, use_bn=use_bn, apply_pooling=False)

        self.contracts = nn.ModuleList(self.contracts)

        if use_dropout:
            self.dropout = nn.Dropout(0.5)

        expand = CreateUpConvBlock(o*8, o*4, o*4, o*4, n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(o*4, o*2, o*2, o*2, n=2, use_bn=use_bn)
        self.expands.append(expand)
         
        expand = CreateUpConvBlock(o*2, o, o, o, n=2, use_bn=use_bn)
        self.expands.append(expand)

        self.expands = nn.ModuleList(self.expands)

    def forward(self, x):
        conv_results = []
        for contract in self.contracts:
            x, conv_result = contract(x)
            conv_results.append(conv_result)

        conv_results = conv_results[::-1]

        x, _ = self.lastContract(x)
        if self.use_dropout:
            x = self.dropout(x)
            
        for expand, conv_result in zip(self.expands, conv_results):
            x = expand(x, conv_result)

        return x

if __name__ == "__main__":
    model=BuildThinUNet(1, 128)
    net_shape = [1, 1, 200, 200, 8]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    dummy_img = torch.rand(net_shape).to(device)
    print("input: ", net_shape)

    x, output = model.contracts[0](dummy_img)
    x, output = model.contracts[1](x)

    print('output:', output.size())
