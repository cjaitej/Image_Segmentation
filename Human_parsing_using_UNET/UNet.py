import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features = [64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        #DownSampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #UpSampling
        for feature in features[::-1]:
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, 2, 2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.last_activation = nn.Softmax(dim=1)
        # self.last_activation = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            if skip_connection.shape != x.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[i+1](concat)
        # x = nn.ReLU()(self.final_conv(x))
        x = nn.Dropout()(x)
        x = self.final_conv(x)
        # x = self.last_activation(x)
        return x

    def decode(mask):
        return torch.argmax(mask, dim=1)

def test():
    x = torch.rand((3, 3, 161, 161))
    model = UNet(in_channels=3, out_channels=24)
    pred = model(x)
    print(pred.shape)
    # assert pred.shape == x.shape

if __name__ == "__main__":
    test()