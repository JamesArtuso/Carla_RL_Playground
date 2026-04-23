import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoderBase(nn.Module):
    def __init__(self, image_encoder, backend):
        super().__init__()
        self.encoder = image_encoder
        self.model = backend
    def forward(self, x):
        raise NotImplementedError

    def forward_sequential(self, x, hidden=None):
        raise NotImplementedError

class ImageEncoderWrapperMLP(ImageEncoderBase):
    def __init__(self, image_encoder, backend):
        super().__init__(image_encoder, backend)

    def forward(self, x, act=None, deterministic=False):
        x_img = x
        #print(f'x_img shape: {x_img.shape}')
        ##try:
        ##    x_state = x['state']
        ##except:
        ##    x_state = None
        encoder_out = self.encoder(x_img)
        ##if(x_state):
        ##    x_in = torch.concat(encoder_out, x_state)
        if(act is None and not deterministic):
            return self.model(encoder_out)
        else:
            return self.model(encoder_out, act, deterministic)
        
        
class ImageEncoderWrapperLSTM(nn.Module):
    """
    Wraps an image encoder + an LSTM backend (actor or critic).

    Rolling usage:
        img: (B, C, H, W) -> z: (B, obs_dim) -> backend.forward(z, ...)

    Sequential usage:
        img_seq: (B, T, C, H, W) -> z_seq: (B, T, obs_dim) -> backend.forward_sequential(z_seq, ..., h0=hidden)
    """
    def __init__(self, image_encoder: nn.Module, backend: nn.Module):
        super().__init__()
        self.encoder = image_encoder
        self.model = backend

    # ---- convenience passthroughs (optional but handy) ----
    def reset_hidden(self, batch_size=1, device=None):
        if hasattr(self.model, "reset_hidden"):
            return self.model.reset_hidden(batch_size=batch_size, device=device)
        raise AttributeError("Backend has no reset_hidden method")

    def init_hidden(self, batch_size=1, device=None):
        if hasattr(self.model, "init_hidden"):
            return self.model.init_hidden(batch_size=batch_size, device=device)
        raise AttributeError("Backend has no init_hidden method")

    # ---- core ----
    def forward(self, x, *args, **kwargs):
        """
        Rolling / single-step forward.

        x: (B, C, H, W)
        Returns: whatever backend.forward returns.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x as (B,C,H,W), got shape {tuple(x.shape)}")

        z = self.encoder(x)  # (B, obs_dim)
        return self.model(z, *args, **kwargs)

    def forward_sequential(self, x, *args, hidden=None, **kwargs):
        """
        Sequential forward for training batches.

        x: (B, T, C, H, W)
        hidden: initial hidden state for the LSTM (h0,c0) or None
        Returns: whatever backend.forward_sequential returns.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected x as (B,T,C,H,W), got shape {tuple(x.shape)}")

        B, T, C, H, W = x.shape

        # Encode per-frame efficiently by flattening batch*time
        x_flat = x.reshape(B * T, C, H, W)      # (B*T, C, H, W)
        z_flat = self.encoder(x_flat)           # (B*T, obs_dim)
        z_seq  = z_flat.reshape(B, T, -1)       # (B, T, obs_dim)

        # Your backends use parameter name h0, so pass hidden as h0.
        # Keep *args/**kwargs to support actor (act_seq) vs critic, etc.
        return self.model.forward_sequential(z_seq, *args, h0=hidden, **kwargs)

class CNN(nn.Module):
    """
    Image encoder matching the table:

    Conv1:  3   -> 32,  k=3, s=2, p=1
    Conv2:  32  -> 64,  k=3, s=2, p=1
    MaxPool:         k=3, s=3, p=0
    Conv3:  64  -> 128, k=3, s=2, p=1
    Conv4:  128 -> 256, k=3, s=2, p=1
    MaxPool:         k=3, s=2, p=0
    FC1:   1024 -> 512
    FC2:    512 -> 256
    FC3:    256 -> 128

    Note: To make the FC input exactly 1024 regardless of input image size,
    this uses AdaptiveAvgPool2d((2, 2)) after the last pooling (256 * 2 * 2 = 1024).
    """
    def __init__(self, output_size = 128):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Ensures flattened size = 256 * 2 * 2 = 1024
        self.to_2x2 = nn.AdaptiveAvgPool2d((2, 2))

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.to_2x2(x)
        x = torch.flatten(x, 1)  # (N, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)          # (N, 128)
        return x