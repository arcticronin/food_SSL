import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):
    # autoencoder roba
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
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


# module that after the convolutional layers,
# upsamples the image. We will use it for colorization
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.decode = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(64, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(32, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 3, kernel_size=3, padding=1),  # Output 3 channels for RGB
#             nn.Tanh(),  # Output values between -1 and 1
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.decode(x)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes: int = 251, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


class ColorizationSqueezeNet(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(ColorizationSqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=3, stride=2, padding=1
            ),  # Change input channels to 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        # Instantiate the decoder
        self.decoder = Decoder()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.decoder(x)
        return x


# Example hyperparameters
# criterion = nn.MSELoss()  # in colorization, mean squared error is mostly used
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


import torch
import torch.nn as nn
import torch.nn.init as init


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






############ Custom Model ############
class StormFire(nn.Module):
    # This model is based on Fire module from SqueezeNet with the addition of BatchNorm 
    # and the change of ReLU to LeakyReLU

    # autoencoder
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        
    ) -> None:
        super().__init__()
        self.inplanes = inplanes

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.LeakyReLU(inplace=True)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)
       
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.LeakyReLU(inplace=True)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes)
        
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.LeakyReLU(inplace=True)
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes)

    def forward(self, x):
        x = self.squeeze_bn(self.squeeze_activation(self.squeeze(x)))
        return torch.cat([
            self.expand1x1_bn(
                self.expand1x1_activation(self.expand1x1(x))
            ),
            self.expand3x3_bn(
                self.expand3x3_activation(self.expand3x3(x)) 
            ),
            ], 1)
            
        


class StormSqueezeNet(nn.Module):
    # This net is a custom version of SqueezeNet: we replaced the Fire module with our custom StormFire module,
    # modified the final convolutional layer into a fully connected layer
    # we used leaky ReLU instead of ReLU
    # we added BatchNorm after each convolutional layer
    # we modified the number of fire modules and the number of filters in each fire module

    def __init__(self, num_classes: int = 251, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            StormFire(64, 16, 64, 64),
            StormFire(128, 16, 64, 64),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            StormFire(128, 32, 128, 128),
            StormFire(256, 32, 128, 128),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            StormFire(256, 48, 192, 192),
            StormFire(384, 64, 256, 256),
            
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            final_conv,
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, self.num_classes),
        )
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier2(x)
        return torch.flatten(x, 1)

