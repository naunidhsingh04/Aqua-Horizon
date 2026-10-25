"""
AQUA HORIZON — 5 Hybrid Deep Learning Architectures

How the hybrids are built
-------------------------
Every network is defined on its own, as a separate class:
  * "first half" models  : CNN, ResNet, U-Net, Attention U-Net   (look at the features / the grid)
  * "second half" models : LSTM (also BiLSTM), Transformer, ConvLSTM   (look at the years in order)

A second-half model takes an optional first-half model as the argument `frontend` and CALLS it at the
start of its own forward():   x = self.frontend(x)   and then does its own job on the result.
A hybrid is therefore just "second-half model + its first-half model plugged in":

    cnn_lstm = LSTMModel(..., frontend=CNNFeatureExtractor(...))        # CNN variable called inside the LSTM
    class CNNLSTM(LSTMModel):  builds the CNN and passes it to LSTMModel.__init__

Work split (half / half):
    CNNLSTM            CNN mixes the 15 features of each year      | LSTM reads the 5 years in order
    CNNTransformer     CNN mixes the 15 features of each year      | Transformer lets the years attend to each other
    ResNetBiLSTM       ResNet extracts deep features of each year  | BiLSTM reads the years forwards and backwards
    UNetConvLSTM       U-Net encodes each 4x4 year grid            | ConvLSTM remembers across years, U-Net decodes
    AttentionUNetLSTM  Attention U-Net summarises each year's grid | LSTM reads the 5 summaries in order

1. U-Net + ConvLSTM
2. CNN + LSTM
3. CNN + Transformer
4. ResNet + BiLSTM
5. Attention U-Net + LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# #############################################################################
# A. FIRST-HALF MODELS  (feature extractors; each works on its own)
# #############################################################################

class CNNFeatureExtractor(nn.Module):
    """CNN. Mixes the input features of every year: (B, T, F) -> (B, T, channels)."""
    def __init__(self, in_features=15, channels=48, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.out_dim = channels

    def forward(self, x):
        # Conv1d wants (B, F, T); permute in, run, permute back so callers keep (B, T, features)
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


class ResidualBlock1D(nn.Module):
    # Output = ReLU(input + transform(input)); the "skip" lets the block only learn a correction
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


class ResNetFeatureExtractor(nn.Module):
    """ResNet. Stem + residual blocks with skip connections: (B, T, F) -> (B, T, channels)."""
    def __init__(self, in_features=15, channels=48, n_blocks=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock1D(channels) for _ in range(n_blocks)])
        self.out_dim = channels

    def forward(self, x):
        return self.blocks(self.stem(x.permute(0, 2, 1))).permute(0, 2, 1)


class UNetEncoder(nn.Module):
    """U-Net encoder. One frame (B, C, H, W) -> (skip (B,16,H,W), features (B,32,H,W))."""
    def __init__(self, in_channels=1):
        super().__init__()
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
        self.skip_channels = 16
        self.out_channels = 32

    def forward(self, x):
        skip = self.enc1(x)
        return skip, self.enc2(skip)


class UNetDecoder(nn.Module):
    """U-Net decoder. Merges deep features with the skip connection and gives one logit:
    (features (B,hidden,H,W), skip (B,16,H,W)) -> (B, 1)."""
    def __init__(self, hidden_dim=32, skip_channels=16):
        super().__init__()
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dim + skip_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 1)
        )

    def forward(self, features, skip):
        return self.head(self.dec1(torch.cat([features, skip], dim=1)))


class UNet(nn.Module):
    """Complete U-Net = UNetEncoder + UNetDecoder. A second-half model (ConvLSTMModel) calls
    encode() on every year and decode() once at the end."""
    def __init__(self, in_channels=1, hidden_dim=32):
        super().__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder(hidden_dim, self.encoder.skip_channels)
        self.out_channels = self.encoder.out_channels

    def encode(self, frame):
        return self.encoder(frame)            # -> (skip, features)

    def decode(self, features, skip):
        return self.decoder(features, skip)   # -> logit (B, 1)

    def forward(self, frame):
        # standalone use on a single frame: encode then decode straight away (no memory)
        skip, feat = self.encoder(frame)
        return self.decoder(feat, skip)


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
        # alpha (0-1) = how important each location of x is, judged using the deeper features g.
        # Output = x * alpha, so unimportant locations are scaled down toward 0.
        theta_x = self.W_x(x)
        phi_g = self.W_g(g)
        f = self.relu(theta_x + phi_g)
        alpha = self.psi(f)
        return x * alpha


class AttentionUNetEncoder(nn.Module):
    """Attention U-Net. Every year's grid (B, T, C, H, W) -> a 48-number summary per year (B, T, 48):
    encode, attention-gate the skip features, average-pool both."""
    def __init__(self, in_channels=1):
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
        self.out_dim = 32 + 16

    def encode_frame(self, x):
        B = x.size(0)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        att_s1 = self.gate(s1, s2)
        p1 = self.pool(att_s1).view(B, -1)
        p2 = self.pool(s2).view(B, -1)
        return torch.cat([p1, p2], dim=-1)

    def forward(self, x):
        return torch.stack([self.encode_frame(x[:, t]) for t in range(x.size(1))], dim=1)


# #############################################################################
# B. SECOND-HALF MODELS  (each works alone, or calls a first-half model via `frontend`)
# #############################################################################

class LSTMModel(nn.Module):
    """LSTM classifier (bidirectional=True makes it a BiLSTM): (B, T, D) -> P(flood) (B,).
    If a first-half model is given as `frontend`, forward() calls it first: x = self.frontend(x)."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, bidirectional=False,
                 lstm_dropout=0.2, head_dropout=0.2, frontend=None):
        super().__init__()
        self.frontend = frontend
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), 32),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if self.frontend is not None:
            x = self.frontend(x)               # first half: CNN / ResNet / Attention U-Net
        out, _ = self.lstm(x)                  # second half: read the years in order
        return torch.sigmoid(self.head(out[:, -1, :])).squeeze(-1)


class TransformerModel(nn.Module):
    """Transformer classifier: (B, T, d_model) -> P(flood) (B,). Adds a learned position per year,
    self-attention over the years, mean over years. If `frontend` is given, forward() calls it first."""
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_len=10,
                 head_dropout=0.15, frontend=None):
        super().__init__()
        self.frontend = frontend
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
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
            nn.Dropout(head_dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if self.frontend is not None:
            x = self.frontend(x)               # first half: CNN
        T = x.size(1)
        h = self.transformer(x + self.pos_encoder[:, :T, :])   # second half: years attend to each other
        return torch.sigmoid(self.head(torch.mean(h, dim=1))).squeeze(-1)


class ConvLSTMCell(nn.Module):
    # An LSTM cell whose gates are convolutions, so it remembers over time while keeping the
    # 2D (4x4) layout. Input/forget/output gates decide what to write, keep and show.
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
        # memory = forget*old_memory + input*new_candidate ; output = out_gate * tanh(memory)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur


class ConvLSTMModel(nn.Module):
    """ConvLSTM over a sequence of 2D grids: (B, T, C, H, W) -> P(flood) (B,).
    It CALLS the U-Net it is given (`unet`): unet.encode() on every year, its own ConvLSTMCell keeps the
    memory across years, then unet.decode() turns the final memory + skip connection into P(flood)."""
    def __init__(self, unet, hidden_dim=32):
        super().__init__()
        self.unet = unet
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(unet.out_channels, hidden_dim)

    def forward(self, x):
        B, T, C, H, W = x.size()
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        last_skip = None
        for t in range(T):
            skip, feat = self.unet.encode(x[:, t])     # first half: U-Net encoder
            h, c = self.cell(feat, h, c)               # second half: ConvLSTM memory
            last_skip = skip
        return torch.sigmoid(self.unet.decode(h, last_skip)).squeeze(-1)


# #############################################################################
# C. HYBRID MODELS  (second-half model with its first-half model plugged in)
# #############################################################################

# 1. U-Net + ConvLSTM ---------------------------------------------------------
class UNetConvLSTM(ConvLSTMModel):
    """ConvLSTM that calls a U-Net inside itself."""
    def __init__(self, in_channels=1, spatial_dim=4, hidden_dim=32):
        self.spatial_dim = spatial_dim
        super().__init__(unet=UNet(in_channels, hidden_dim), hidden_dim=hidden_dim)


# 2. CNN + LSTM ---------------------------------------------------------------
class CNNLSTM(LSTMModel):
    """LSTM that calls a CNN inside itself."""
    def __init__(self, in_features=15, cnn_channels=48, lstm_hidden=64, num_layers=2):
        cnn = CNNFeatureExtractor(in_features, cnn_channels)
        super().__init__(input_size=cnn.out_dim, hidden_size=lstm_hidden, num_layers=num_layers, frontend=cnn)


# 3. CNN + Transformer --------------------------------------------------------
class CNNTransformer(TransformerModel):
    """Transformer that calls a CNN inside itself."""
    def __init__(self, in_features=15, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        cnn = CNNFeatureExtractor(in_features, d_model, dropout=0.15)
        super().__init__(d_model, nhead, num_layers, dim_feedforward, frontend=cnn)


# 4. ResNet + BiLSTM ----------------------------------------------------------
class ResNetBiLSTM(LSTMModel):
    """BiLSTM that calls a ResNet inside itself."""
    def __init__(self, in_features=15, res_channels=48, lstm_hidden=40):
        resnet = ResNetFeatureExtractor(in_features, res_channels, n_blocks=2)
        super().__init__(input_size=resnet.out_dim, hidden_size=lstm_hidden, num_layers=2,
                         bidirectional=True, frontend=resnet)


# 5. Attention U-Net + LSTM ---------------------------------------------------
class AttentionUNetLSTM(LSTMModel):
    """LSTM that calls an Attention U-Net inside itself."""
    def __init__(self, in_channels=1, hidden_dim=32, lstm_hidden=48):
        attn_unet = AttentionUNetEncoder(in_channels)
        super().__init__(input_size=attn_unet.out_dim, hidden_size=lstm_hidden, num_layers=1,
                         head_dropout=0.0, frontend=attn_unet)


if __name__ == '__main__':
    # Self-test: (1) each model alone, (2) each hybrid = second-half model with its first-half model inside.
    torch.manual_seed(42)
    seq = torch.randn(4, 5, 15)        # 4 districts x 5 years x 15 features
    grid = torch.randn(4, 5, 1, 4, 4)  # the same data as 4x4 grids, for the U-Net models

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    print("=" * 86)
    print("PART 1 - EACH MODEL ON ITS OWN")
    print("=" * 86)
    alone = [
        ("CNN",              CNNFeatureExtractor(),        seq,           (4, 5, 48)),
        ("ResNet",           ResNetFeatureExtractor(),     seq,           (4, 5, 48)),
        ("Attention U-Net",  AttentionUNetEncoder(),       grid,          (4, 5, 48)),
        ("U-Net (1 frame)",  UNet(),                       grid[:, 0],    (4, 1)),
        ("LSTM",             LSTMModel(15, 64),            seq,           (4,)),
        ("BiLSTM",           LSTMModel(15, 40, bidirectional=True), seq,  (4,)),
        ("Transformer",      TransformerModel(64),         torch.randn(4, 5, 64), (4,)),
    ]
    for name, model, x, expected in alone:
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert tuple(out.shape) == expected, f"{name}: expected {expected}, got {tuple(out.shape)}"
        print(f"{name:<18} | {n_params(model):>8,} params | {str(tuple(x.shape)):<16} -> {tuple(out.shape)}")

    print()
    print("=" * 86)
    print("PART 2 - HYBRIDS (the second-half model calls the first-half model inside its forward)")
    print("=" * 86)
    tests = [
        ("U-Net + ConvLSTM",       UNetConvLSTM(),      grid, "unet"),
        ("CNN + LSTM",             CNNLSTM(),           seq,  "frontend"),
        ("CNN + Transformer",      CNNTransformer(),    seq,  "frontend"),
        ("ResNet + BiLSTM",        ResNetBiLSTM(),      seq,  "frontend"),
        ("Attention U-Net + LSTM", AttentionUNetLSTM(), grid, "frontend"),
    ]
    print(f"{'Model':<24} | {'Params':>8} | {'First half called inside':<28} | P(flood) for 4 districts")
    print("-" * 86)
    for name, model, x, attr in tests:
        first_half = type(getattr(model, attr)).__name__
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4,) and bool(((out >= 0) & (out <= 1)).all()), "output must be 4 probabilities in [0, 1]"
        print(f"{name:<24} | {n_params(model):>8,} | {first_half:<28} | {[round(float(v), 3) for v in out]}")
    print("-" * 86)
    print("All 5 hybrids are built by plugging a first-half model into a second-half model and give valid probabilities.")
