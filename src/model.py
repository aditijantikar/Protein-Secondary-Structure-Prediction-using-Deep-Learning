import torch
import torch.nn as nn

class CNNTransformerPSSP(nn.Module):
    def __init__(self, input_dim=21, cnn_out_dim=128, transformer_dim=256, num_heads=4,
                 num_layers=2, dropout=0.3, num_classes=8, sequence_len=700):
        super(CNNTransformerPSSP, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_dim, kernel_size=11, padding=5),
            nn.BatchNorm1d(cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_len, cnn_out_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(cnn_out_dim, num_classes)

    def forward(self, x, mask=None):
        # x: (B, L, input_dim)
        x = x.permute(0, 2, 1)               # -> (B, input_dim, L)
        x = self.cnn(x)                     # -> (B, cnn_out_dim, L)
        x = x.permute(0, 2, 1)              # -> (B, L, cnn_out_dim)

        x = x + self.pos_embedding[:, :x.size(1), :]

        if mask is not None:
            # mask: (B, L) with True=valid token; Transformer wants key_padding_mask where True=PAD to ignore
            key_padding_mask = ~mask.bool()  # invert: True where padding
            x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.transformer_encoder(x)

        logits = self.fc(x)  # (B, L, num_classes)
        return logits