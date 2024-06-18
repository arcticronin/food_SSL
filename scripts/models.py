import torch
import torch.nn as nn
import torch.nn.init as init


##############################################################################        
#                                                                            #
#                                Custom Models                               #
#                                                                            #
##############################################################################
class StormModel(nn.Module):
    # This net is a custom version of SqueezeNet: we replaced the Fire module with our custom FireStorm module,
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

            FireStorm(64, 16, 64, 64),
            FireStorm(128, 16, 64, 64),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireStorm(128, 32, 128, 128),
            FireStorm(256, 32, 128, 128),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireStorm(256, 48, 192, 192),
            FireStorm(384, 64, 256, 256),
            
        )


        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, self.num_classes),
    )
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)






class FireStorm(nn.Module):
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
            



class StormColorModel(nn.Module):
    # This net is a custom version of SqueezeNet: we replaced the Fire module with our custom FireStorm module,
    # modified the final convolutional layer into a fully connected layer
    # we used leaky ReLU instead of ReLU
    # we added BatchNorm after each convolutional layer
    # we modified the number of fire modules and the number of filters in each fire module

    def __init__(self, num_classes: int = 251, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireStorm(64, 16, 64, 64),
            FireStorm(128, 16, 64, 64),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireStorm(128, 32, 128, 128),
            FireStorm(256, 32, 128, 128),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireStorm(256, 48, 192, 192),
            FireStorm(384, 64, 256, 256),
            
        )

     
        self.decoder = nn.Sequential(

            FireStorm(512, 64, 128, 128),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),

            FireStorm(128, 32, 32, 32),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),

            FireStorm(32, 16, 16, 16),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.features(x)
        return self.decoder(encoded)




#################################################################################################################

    
class StormModel2(nn.Module):
    # version with more FireStorm modules and parameters

    def __init__(self, num_classes: int = 251, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireStorm(64, 16, 64, 64),
            FireStorm(128, 16, 64, 64),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireStorm(128, 32, 128, 128),
            FireStorm(256, 32, 128, 128),
            FireStorm(256, 48, 192, 192),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
      
            FireStorm(384, 64, 192, 192),
            FireStorm(384, 64, 256, 256),  # added module
            
        )