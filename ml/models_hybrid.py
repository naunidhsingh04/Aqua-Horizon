"""
AQUA HORIZON — 5 Hybrid Deep Learning Architectures
1. U-Net + ConvLSTM
2. CNN + LSTM
3. CNN + Transformer
4. ResNet + BiLSTM
5. Attention U-Net + LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. U-Net + ConvLSTM
# =============================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur


class UNetConvLSTM(nn.Module):
    """U-Net spatial encoder-decoder fused with ConvLSTM temporal recurrent cell."""
    def __init__(self, in_channels=1, spatial_dim=4, hidden_dim=32):
        super().__init__()
        self.spatial_dim = spatial_dim
        # Spatial Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # ConvLSTM Bottleneck
        self.conv_lstm = ConvLSTMCell(32, hidden_dim)
        
        # Spatial Decoder with Skip Connection
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dim + 16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        h = torch.zeros(B, 32, H, W, device=x.device)
        c = torch.zeros(B, 32, H, W, device=x.device)
        
        last_skip = None
        for t in range(T):
            xt = x[:, t]  # (B, C, H, W)
            skip = self.enc1(xt)
            feat = self.enc2(skip)
            h, c = self.conv_lstm(feat, h, c)
            last_skip = skip
            
        dec = self.dec1(torch.cat([h, last_skip], dim=1))
        out = torch.sigmoid(self.head(dec)).squeeze(-1)
        return out


# =============================================================================
# 2. CNN + LSTM
# =============================================================================

class CNNLSTM(nn.Module):
    """1D-CNN temporal feature extractor + 2-layer LSTM sequence modeling."""
    def __init__(self, in_features=15, cnn_channels=48, lstm_hidden=64, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_features, cnn_channels, kernel_size=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (B, T, F) -> permute for 1D CNN: (B*T, F, 1) or (B, F, T)
        B, T, F = x.size()
        xt = x.permute(0, 2, 1)  # (B, F, T)
        cnn_out = self.cnn(xt)  # (B, cnn_channels, T)
        lstm_in = cnn_out.permute(0, 2, 1)  # (B, T, cnn_channels)
        lstm_out, _ = self.lstm(lstm_in)
        last_hidden = lstm_out[:, -1, :]  # (B, lstm_hidden)
        out = torch.sigmoid(self.head(last_hidden)).squeeze(-1)
        return out


# =============================================================================
# 3. CNN + Transformer
# =============================================================================

class CNNTransformer(nn.Module):
    """CNN feature projector + Multi-Head Self-Attention Transformer encoder."""
    def __init__(self, in_features=15, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (B, T, F)
        B, T, F = x.size()
        h = self.proj(x) + self.pos_encoder[:, :T, :]
        trans_out = self.transformer(h)  # (B, T, d_model)
        pooled = torch.mean(trans_out, dim=1)  # Temporal mean pooling
        out = torch.sigmoid(self.head(pooled)).squeeze(-1)
        return out


# =============================================================================
# 4. ResNet + BiLSTM
# =============================================================================

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNetBiLSTM(nn.Module):
    """Deep Residual Blocks with skip connections + Bidirectional LSTM."""
    def __init__(self, in_features=15, res_channels=48, lstm_hidden=40):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, res_channels, kernel_size=1),
            nn.BatchNorm1d(res_channels),
            nn.ReLU()
        )
        self.res1 = ResidualBlock1D(res_channels)
        self.res2 = ResidualBlock1D(res_channels)
        
        # BiLSTM processes temporal sequences forwards and backwards
        self.bilstm = nn.LSTM(
            input_size=res_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (B, T, F)
        B, T, F = x.size()
        xt = x.permute(0, 2, 1)  # (B, F, T)
        h = self.stem(xt)
        h = self.res1(h)
        h = self.res2(h)
        bilstm_in = h.permute(0, 2, 1)  # (B, T, res_channels)
        bilstm_out, _ = self.bilstm(bilstm_in)
        last_step = bilstm_out[:, -1, :]  # (B, lstm_hidden*2)
        out = torch.sigmoid(self.head(last_step)).squeeze(-1)
        return out


# =============================================================================
# 5. Attention U-Net + LSTM
# =============================================================================

class AttentionGate(nn.Module):
    """Additive attention gate to filter spatial hydrological activations."""
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, x, g):
        theta_x = self.W_x(x)
        phi_g = self.W_g(g)
        f = self.relu(theta_x + phi_g)
        alpha = self.psi(f)
        return x * alpha


class AttentionUNetLSTM(nn.Module):
    """Attention-gated U-Net feature backbone + LSTM recurrent predictor."""
    def __init__(self, in_channels=1, hidden_dim=32, lstm_hidden=48):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.gate = AttentionGate(in_channels=16, gating_channels=32, inter_channels=16)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(
            input_size=32 + 16,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        time_feats = []
        for t in range(T):
            xt = x[:, t]
            s1 = self.enc1(xt)
            s2 = self.enc2(s1)
            att_s1 = self.gate(s1, s2)
            p1 = self.pool(att_s1).view(B, -1)
            p2 = self.pool(s2).view(B, -1)
            f_t = torch.cat([p1, p2], dim=-1)  # (B, 16+32)
            time_feats.append(f_t.unsqueeze(1))
            
        seq = torch.cat(time_feats, dim=1)  # (B, T, 48)
        lstm_out, _ = self.lstm(seq)
        out = torch.sigmoid(self.head(lstm_out[:, -1, :])).squeeze(-1)
        return out