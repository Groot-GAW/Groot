import torch.nn as nn


class MGCN(nn.Module):
    """
    MGCN: Modified Gated Convolutional Neural Network
    input: audio features -> (B, C, L)
    output: result -> (B, C, L)
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=0):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv1d.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.mask.weight, mode='fan_in', nonlinearity='relu')

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, fts):
        conv = self.conv1d(fts)
        mask = self.bn(self.mask(fts))
        gated = self.sigmoid(mask)

        res = conv * gated

        return res


class WMEncoder(nn.Module):
    """
    WMEncoder: watermark encoder, transforming watermark into the latent variable
    parameters: @watermark_length: length of watermark,
                @gaussian_length: length of gaussian latent variable,
                @audio_length: audio duration
    input: watermark -> (B, L_wm)
    output: latent variable sigma -> (B, L_latent)
    """
    def __init__(self, watermark_length, gaussian_length=22272, audio_length=1):
        super().__init__()

        self.sec_len = watermark_length
        self.latent_length = gaussian_length * audio_length

        self.fc1 = nn.Linear(watermark_length, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.fc5 = nn.Linear(4096, 8192)
        self.fc6 = nn.Linear(8192, self.latent_length)
        self.act = nn.ReLU()

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, watermark):
        fc1 = self.act(self.fc1(watermark))
        fc2 = self.act(self.fc2(fc1))
        fc3 = self.act(self.fc3(fc2))
        fc4 = self.act(self.fc4(fc3))
        fc5 = self.act(self.fc5(fc4))
        latent_sigma = self.act(self.fc6(fc5))

        return latent_sigma


class WMDecoder(nn.Module):
    """
    WMDecoder: watermark decoder
    parameter: @watermark_length: length of watermark
    input: watermarked -> (B, C, L)
    output: extracted_watermark -> (B, L_wm)
    """
    def __init__(self, watermark_length):
        super().__init__()

        # ConvBlock
        self.decoder = nn.Sequential(
            MGCN(1, 32, kernel_size=3, stride=2, padding=1),
            MGCN(32, 32, kernel_size=3, stride=2, padding=1),
            MGCN(32, 64, kernel_size=3, stride=2, padding=1),
            MGCN(64, 64, kernel_size=3, stride=2, padding=1),
            MGCN(64, 64, kernel_size=3, stride=2, padding=1),
            MGCN(64, 128, kernel_size=3, stride=2, padding=1),
            MGCN(128, 256, kernel_size=3, stride=2, padding=1)
        )
        # Dense Block
        self.dense = nn.Sequential(
            nn.Linear(256*174, 512),  # 174
            nn.ReLU(),
            nn.Linear(512, watermark_length)
        )

    def forward(self, watermarked):
        decoded = self.decoder(watermarked)
        fts = decoded.view(-1, 256*174)
        extracted_watermark = self.dense(fts)

        return extracted_watermark
