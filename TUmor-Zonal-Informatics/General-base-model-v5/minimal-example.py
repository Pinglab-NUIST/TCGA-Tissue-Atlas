import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=3):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, model_path=None):
        super(Unet, self).__init__()
        self.vgg = vgg16(pretrained=pretrained)
        del self.vgg.classifier
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]  # 每一次up
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64  （最终有效特征层）
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        #   official Coord Attention
        self.cablock1 = CoordAtt(512, 512)
        self.cablock2 = CoordAtt(256, 256)
        self.cablock3 = CoordAtt(128, 128)
        self.cablock4 = CoordAtt(64, 64)

        #   Final conv
        self.final = nn.Sequential(nn.Conv2d(out_filters[0], num_classes, 1),
                                   nn.Softmax2d())

        self.backbone_type = 'vgg'
        self.output_downsample = 1

        #   载入已有权重
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))

    def forward(self, inputs):
        feat1 = self.vgg.features[:4](inputs)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        up4 = self.up_concat4(self.cablock1(feat4), feat5)
        up3 = self.up_concat3(self.cablock2(feat3), up4)
        up2 = self.up_concat2(self.cablock3(feat2), up3)
        up1 = self.up_concat1(self.cablock4(feat1), up2)
        final = self.final(up1)

        return final

    #   也不知道怎么调用它，算了不管了
    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


#   CA block的官方实现
#   https://github.com/Andrew-Qibin/CoordAttention/blob/main/coordatt.py
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


if __name__ == "__main__":
    ''' A minimal example using trained model checkpoint'''
    image_batch = torch.rand(size=(1, 3, 512, 512))
    model_path = r"general_base_model_0.5_v5@108.pth"
    model = Unet(num_classes=15, model_path=model_path)
    output = model(image_batch)
    print(output.shape)

    '''Minimal examples of processing real images'''
    import numpy as np
    from PIL import Image
    from torchvision.transforms import Compose, ToTensor, Normalize
    import matplotlib.pyplot as plt

    preprocess = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    cmap = plt.get_cmap("tab20")

    for image_file in ["TCGA-2J-AAB1-01Z-00-DX1.F3B4818F-9C3B-4C66-8241-0570B2873EC9_x96349_y40368.jpg",
                       "TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291_x56783_y36162.jpg",
                       "TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4_x72283_y30796.jpg"]:
        #   forwrad
        image=Image.open(image_file)
        image_data=np.array(image)
        image_batch = preprocess(image).unsqueeze(0)
        pred = model(image_batch)[0].detach().cpu().numpy()
        segmap = np.argmax(pred, axis=0)

        #   visualize
        mask_color = cmap(segmap % cmap.N)[..., :3]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(image_data)
        ax[0].set_title("RGB")
        ax[0].axis("off")
        ax[1].imshow(segmap)
        ax[1].set_title("Mask")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()

