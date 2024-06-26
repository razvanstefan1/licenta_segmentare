# 2D-Unet Model taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
import torch
import torch.nn as nn

class unetConv2(nn.Module):                                            #n=nr de conv layers in acest bloc, ks = kernel_size
    def __init__(self, input_channel_size, output_channel_size, use_batchnorm, n=3, kernel_size=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if use_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(input_channel_size, output_channel_size, kernel_size, s, p),
                                     nn.BatchNorm2d(output_channel_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                input_channel_size = output_channel_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(input_channel_size, output_channel_size, kernel_size, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                input_channel_size = output_channel_size


    def forward(self, inputs): #aplicam fiecare convolutional layer in mod secvential, pe input
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x
class unetUp(nn.Module): #upsampling block
    def __init__(self, input_channel_size, output_channel_size, use_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(input_channel_size, output_channel_size, False)
        if use_deconv:
            self.up = nn.ConvTranspose2d(output_channel_size, output_channel_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)



    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
class UNet(nn.Module):

    def __init__(self, in_channels, n_classes, channels=64, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.channels = channels
        self.n_classes=n_classes

        # downsampling
        self.conv1 = unetConv2(self.in_channels, self.channels, self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(self.channels, self.channels, self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(self.channels*2, self.channels, self.is_deconv)
        self.up_concat3 = unetUp(self.channels*2, self.channels, self.is_deconv)
        self.up_concat2 = unetUp(self.channels*2, self.channels, self.is_deconv)
        self.up_concat1 = unetUp(self.channels*2, self.channels, self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(self.channels, self.n_classes, 3, padding=1)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(center, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)

        output = self.outconv1(up1)

        return output
