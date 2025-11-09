import torch.nn as nn

class ACPEAdapter(nn.Module):
    def __init__(self, D, kt, ks, use_mix):
        super().__init__()
        self.use_mix = use_mix
        self.pos_conv = nn.Conv2d(
            in_channels=D,
            out_channels=D,
            kernel_size=(ks, kt),
            stride=(1, 1),
            # Handles even/odd kernel sizes while preserving output shape
            padding='same',
            groups=D  # depthwise
        )

        if self.use_mix:
            self.mix = nn.Linear(D, D)

    def forward(self, x):
        """
        x: (batch_size, n_temporal_channels, n_spatial_channels, D)
        """

        batch_size, n_temporal_channels, n_spatial_channels, D = x.shape
        # Permute to Conv2d expected shape: (batch_size, D, n_spatial_channels, n_temporal_channels)
        x_perm = x.permute(0, 3, 2, 1)  # (batch_size, D, n_spatial_channels, n_temporal_channels)

        # Apply depthwise asymmetric conv
        pos = self.pos_conv(x_perm)  # (batch_size, D, n_spatial_channels, n_temporal_channels)

        # Permute back to original format
        pos = pos.permute(0, 3, 2, 1)  # (batch_size, n_temporal_channels, n_spatial_channels, D)
        pos = pos.reshape(batch_size, n_temporal_channels * n_spatial_channels, D)
        if self.use_mix:
            pos = self.mix(pos)

        return pos