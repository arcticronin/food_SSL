{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"},"kaggle":{"accelerator":"none","dataSources":[],"isInternetEnabled":true,"language":"python","sourceType":"script","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code]\nimport torch\nimport torch.nn as nn\nimport torch.nn.init as init\n\n\nclass Fire(nn.Module):\n    # autoencoder roba\n    def __init__(\n        self,\n        inplanes: int,\n        squeeze_planes: int,\n        expand1x1_planes: int,\n        expand3x3_planes: int,\n    ) -> None:\n        super().__init__()\n        self.inplanes = inplanes\n        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)\n        self.squeeze_activation = nn.ReLU(inplace=True)\n        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)\n        self.expand1x1_activation = nn.ReLU(inplace=True)\n        self.expand3x3 = nn.Conv2d(\n            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1\n        )\n        self.expand3x3_activation = nn.ReLU(inplace=True)\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        x = self.squeeze_activation(self.squeeze(x))\n        return torch.cat(\n            [\n                self.expand1x1_activation(self.expand1x1(x)),\n                self.expand3x3_activation(self.expand3x3(x)),\n            ],\n            1,\n        )\n\n\n# module that after the convolutional layers,\n# upsamples the image. We will use it for colorization\n# class Decoder(nn.Module):\n#     def __init__(self):\n#         super(Decoder, self).__init__()\n#         self.decode = nn.Sequential(\n#             nn.Conv2d(512, 256, kernel_size=3, padding=1),\n#             nn.ReLU(inplace=True),\n#             nn.Upsample(scale_factor=2),\n#             nn.Conv2d(256, 128, kernel_size=3, padding=1),\n#             nn.ReLU(inplace=True),\n#             nn.Upsample(scale_factor=2),\n#             nn.Conv2d(128, 64, kernel_size=3, padding=1),\n#             nn.ReLU(inplace=True),\n#             nn.Upsample(scale_factor=2),\n#             nn.Conv2d(64, 32, kernel_size=3, padding=1),\n#             nn.ReLU(inplace=True),\n#             nn.Upsample(scale_factor=2),\n#             nn.Conv2d(32, 16, kernel_size=3, padding=1),\n#             nn.ReLU(inplace=True),\n#             nn.Conv2d(16, 3, kernel_size=3, padding=1),  # Output 3 channels for RGB\n#             nn.Tanh(),  # Output values between -1 and 1\n#         )\n\n#     def forward(self, x: torch.Tensor) -> torch.Tensor:\n#         return self.decode(x)\n\n\nclass SqueezeNet(nn.Module):\n    def __init__(self, num_classes: int = 251, dropout: float = 0.5) -> None:\n        super().__init__()\n        self.num_classes = num_classes\n        self.features = nn.Sequential(\n            nn.Conv2d(3, 64, kernel_size=3, stride=2),\n            nn.ReLU(inplace=True),\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n            Fire(64, 16, 64, 64),\n            Fire(128, 16, 64, 64),\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n            Fire(128, 32, 128, 128),\n            Fire(256, 32, 128, 128),\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n            Fire(256, 48, 192, 192),\n            Fire(384, 48, 192, 192),\n            Fire(384, 64, 256, 256),\n            Fire(512, 64, 256, 256),\n        )\n\n        # Final convolution is initialized differently from the rest\n        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)\n        self.classifier = nn.Sequential(\n            nn.Dropout(p=dropout),\n            final_conv,\n            nn.ReLU(inplace=True),\n            nn.AdaptiveAvgPool2d((1, 1)),\n        )\n\n        for m in self.modules():\n            if isinstance(m, nn.Conv2d):\n                if m is final_conv:\n                    init.normal_(m.weight, mean=0.0, std=0.01)\n                else:\n                    init.kaiming_uniform_(m.weight)\n                if m.bias is not None:\n                    init.constant_(m.bias, 0)\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        x = self.features(x)\n        x = self.classifier(x)\n        return torch.flatten(x, 1)\n\n\nclass ColorizationSqueezeNet(nn.Module):\n    def __init__(self, dropout: float = 0.5) -> None:\n        super(ColorizationSqueezeNet, self).__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(\n                1, 64, kernel_size=3, stride=2, padding=1\n            ),  # Change input channels to 1\n            nn.ReLU(inplace=True),\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n            Fire(64, 16, 64, 64),\n            Fire(128, 16, 64, 64),\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n            Fire(128, 32, 128, 128),\n            Fire(256, 32, 128, 128),\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n            Fire(256, 48, 192, 192),\n            Fire(384, 48, 192, 192),\n            Fire(384, 64, 256, 256),\n            Fire(512, 64, 256, 256),\n        )\n        # Instantiate the decoder\n        self.decoder = Decoder()\n\n        self._initialize_weights()\n\n    def _initialize_weights(self):\n        for m in self.modules():\n            if isinstance(m, nn.Conv2d):\n                init.kaiming_uniform_(m.weight)\n                if m.bias is not None:\n                    init.constant_(m.bias, 0)\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        x = self.features(x)\n        x = self.decoder(x)\n        return x\n\n\n# Example hyperparameters\n# criterion = nn.MSELoss()  # in colorization, mean squared error is mostly used\n# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)\n\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.init as init\n\n\nclass FireDecoder(nn.Module):\n    def __init__(self, inplanes: int, squeeze_planes: int, expand_planes: int):\n        super(FireDecoder, self).__init__()\n        self.inplanes = inplanes\n        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)\n        self.squeeze_activation = nn.ReLU(inplace=True)\n        self.expand = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, padding=1)\n        self.expand_activation = nn.ReLU(inplace=True)\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        x = self.squeeze_activation(self.squeeze(x))\n        x = self.expand_activation(self.expand(x))\n        return x\n\n\nclass Decoder(nn.Module):\n    def __init__(self):\n        super(Decoder, self).__init__()\n        self.decode = nn.Sequential(\n            nn.Conv2d(512, 256, kernel_size=1),\n            nn.ReLU(inplace=True),\n            nn.Upsample(scale_factor=2),\n            FireDecoder(256, 64, 128),\n            nn.Upsample(scale_factor=2),\n            FireDecoder(128, 32, 64),\n            nn.Upsample(scale_factor=2),\n            FireDecoder(64, 16, 32),\n            nn.Upsample(scale_factor=2),\n            FireDecoder(32, 8, 16),\n            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # Output 3 channels for RGB\n            nn.Tanh(),  # Output values between -1 and 1\n        )\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.decode(x)\n\n\n\n\n\n\n##############################################################################        \n#                                                                            #\n#                                Custom Models                               #\n#                                                                            #\n##############################################################################\n\n\nclass StormModel(nn.Module):\n    # This net is a custom version of SqueezeNet: we replaced the Fire module with our custom FireStorm module,\n    # modified the final convolutional layer into a fully connected layer\n    # we used leaky ReLU instead of ReLU\n    # we added BatchNorm after each convolutional layer\n    # we modified the number of fire modules and the number of filters in each fire module\n\n    def __init__(self, num_classes: int = 251, dropout: float = 0.5) -> None:\n        super().__init__()\n        self.num_classes = num_classes\n        self.features = nn.Sequential(\n            \n            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding = 1),\n            nn.LeakyReLU(inplace=True),\n\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n\n            FireStorm(64, 16, 64, 64),\n            FireStorm(128, 16, 64, 64),\n\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n\n            FireStorm(128, 32, 128, 128),\n            FireStorm(256, 32, 128, 128),\n\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n\n            FireStorm(256, 48, 192, 192),\n            FireStorm(384, 64, 256, 256),\n            \n        )\n\n\n        self.classifier = nn.Sequential(\n            nn.AdaptiveAvgPool2d((1, 1)),\n            nn.Flatten(),\n            nn.Linear(512, 512),\n            nn.LeakyReLU(inplace=True),\n            nn.Dropout(p=dropout),\n            nn.Linear(512, self.num_classes),\n        )\n       \n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        x = self.features(x)\n        x = self.classifier(x)\n        return torch.flatten(x, 1)\n\n\n\n\nclass FireStorm(nn.Module):\n    # This model is based on Fire module from SqueezeNet with the addition of BatchNorm \n    # and the change of ReLU to LeakyReLU\n\n    # autoencoder\n    def __init__(\n        self,\n        inplanes: int,\n        squeeze_planes: int,\n        expand1x1_planes: int,\n        expand3x3_planes: int,\n        \n    ) -> None:\n        super().__init__()\n        self.inplanes = inplanes\n\n        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)\n        self.squeeze_activation = nn.LeakyReLU(inplace=True)\n        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)\n       \n        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)\n        self.expand1x1_activation = nn.LeakyReLU(inplace=True)\n        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes)\n        \n        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)\n        self.expand3x3_activation = nn.LeakyReLU(inplace=True)\n        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes)\n\n    def forward(self, x):\n        x = self.squeeze_bn(self.squeeze_activation(self.squeeze(x)))\n        return torch.cat([\n            self.expand1x1_bn(\n                self.expand1x1_activation(self.expand1x1(x))\n            ),\n            self.expand3x3_bn(\n                self.expand3x3_activation(self.expand3x3(x)) \n            ),\n            ], 1)\n            \n\n\n\nclass StormColorModel(nn.Module):\n    # This net is a custom version of SqueezeNet: we replaced the Fire module with our custom FireStorm module,\n    # modified the final convolutional layer into a fully connected layer\n    # we used leaky ReLU instead of ReLU\n    # we added BatchNorm after each convolutional layer\n    # we modified the number of fire modules and the number of filters in each fire module\n\n    def __init__(self, num_classes: int = 251, dropout: float = 0.5) -> None:\n        super().__init__()\n        self.num_classes = num_classes\n\n        self.features = nn.Sequential(\n            \n            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),\n            nn.LeakyReLU(inplace=True),\n\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n\n            FireStorm(64, 16, 64, 64),\n            FireStorm(128, 16, 64, 64),\n\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n\n            FireStorm(128, 32, 128, 128),\n            FireStorm(256, 32, 128, 128),\n\n            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),\n\n            FireStorm(256, 48, 192, 192),\n            FireStorm(384, 64, 256, 256),\n            \n        )\n\n     \n        self.decoder = nn.Sequential(\n\n            FireStorm(512, 64, 128, 128),\n\n            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),\n            nn.LeakyReLU(inplace=True),\n            nn.BatchNorm2d(128),\n\n            FireStorm(128, 32, 32, 32),\n\n            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),\n            nn.LeakyReLU(inplace=True),\n            nn.BatchNorm2d(32),\n\n            FireStorm(32, 16, 16, 16),\n\n            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),\n            nn.LeakyReLU(inplace=True),\n            nn.BatchNorm2d(16),\n\n            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),\n            nn.Sigmoid()\n        )\n\n\n\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        encoded = self.features(x)\n        return self.decoder(encoded)\n","metadata":{"_uuid":"c162b031-3c4c-4855-8e24-5aba8329c779","_cell_guid":"d6ca73e2-2392-48d1-8566-031d1f517f97","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}