import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):
    # autoencoder roba
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand_planes: int,
    ) -> None:
        super().__init__()
        self.inplanes = inplanes

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand_planes, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            1,
        )


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64),
            Fire(128, 16, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128),
            Fire(256, 32, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192),
            Fire(384, 48, 192),
            Fire(384, 64, 256),
            Fire(512, 64, 256),
        )


class Encoder_greyscale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64),
            Fire(128, 16, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128),
            Fire(256, 32, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192),
            Fire(384, 48, 192),
            Fire(384, 64, 256),
            Fire(512, 64, 256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes: int = 251, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = Encoder()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Ensures the output is (N, C, 1, 1)
            nn.Flatten(),  # Flatten the output to (N, C)
            nn.Linear(512, 350),  # Linear layer to get the desired feature dimension
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(350, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


class FireDecoder(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand_planes: int):
        super(FireDecoder, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        x = self.expand_activation(self.expand(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            FireDecoder(256, 64, 128),
            nn.Upsample(scale_factor=2),
            FireDecoder(128, 32, 64),
            nn.Upsample(scale_factor=2),
            FireDecoder(64, 16, 32),
            nn.Upsample(scale_factor=2),
            FireDecoder(32, 8, 16),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # Output 3 channels for RGB
            nn.Tanh(),  # Output values between -1 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)


class ColorizationSqueezeNet(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(ColorizationSqueezeNet, self).__init__()
        self.encoder = Encoder_greyscale()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Example hyperparameters
# criterion = nn.MSELoss()  # in colorization, mean squared error is mostly used
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
