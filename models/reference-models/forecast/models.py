"""Window encoders with grouped direct forecast heads."""
import torch
from torch import nn
from torch.nn import functional as F


TARGET_GROUPS = ((0, 1), tuple(range(2, 8)), tuple(range(8, 11)), (11, 12, 13))
OUTPUT_GROUPS = {'O0': (0,), 'O1': (0, 1), 'O2': (0, 2, 3), 'O3': (0, 1, 2, 3)}


def architecture(spec, channels):
    """Describe the architecture from the frozen run parameters."""
    family, large = spec['family'], spec['size'] == 'large'
    length = round(spec['L'] / .1)
    detail = dict(family=family, input_channels=channels, input_steps=length,
                  horizons=spec['horizons'], output=spec['output'])
    if family in ('gru', 'gru16'):
        detail.update(hidden=16 if family == 'gru16' else 256 if large else 128,
                      layers=1 if family == 'gru16' else 3 if large else 2,
                      dropout=0. if family == 'gru16' else .15 if large else .1,
                      direction='unidirectional')
    if family == 'tcn':
        # Two convolutions with k=3 per block contribute 4*d history samples.
        dilations = [1, 2, 4, 8][:next(i for i in range(1, 5) if 1 + 4 * (2**i - 1) >= length)]
        field = 1 + 4 * sum(dilations)
        detail.update(hidden=256 if large else 128, layers=len(dilations), kernel=3,
                      dilations=dilations, receptive_field_steps=field,
                      receptive_field_seconds=(field - 1) * .1,
                      available_field_steps=min(length, field), dropout=.15 if large else .1,
                      padding='causal left')
    if family in ('transformer', 'patch'):
        detail.update(hidden=256 if large else 128, layers=6 if large else 3, heads=8,
                      feed_forward=1024 if large else 512, dropout=.15 if large else .1,
                      position='learned fixed-grid ordinal with 0.1 s spacing',
                      attention='full within a single available past window')
    if family == 'patch':
        patch = min(5, length)
        stride = max(1, patch // 2)
        starts = sorted(set([*range(0, length - patch + 1, stride), length - patch]))
        detail.update(patch_size=patch, patch_stride=stride, patch_starts=starts,
                      patch_count=len(starts), channel_fusion='joint linear projection of all channels inside each patch')
    return detail


class ResidualBlock(nn.Module):
    def __init__(self, width, kernel, dilation, dropout):
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(width, width, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, kernel, dilation=dilation)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        current = self.conv1(F.pad(value, (self.left_pad, 0)))
        current = self.dropout(F.gelu(self.norm1(current.transpose(1, 2)))).transpose(1, 2)
        current = self.conv2(F.pad(current, (self.left_pad, 0)))
        current = self.dropout(self.norm2(current.transpose(1, 2))).transpose(1, 2)
        return F.gelu(value + current)


class Forecaster(nn.Module):
    def __init__(self, detail):
        super().__init__()
        self.family = family = detail['family']
        channels, width = detail['input_channels'], detail['hidden']
        self.horizon_count = len(detail['horizons'])
        if family in ('gru', 'gru16'):
            self.encoder = nn.GRU(channels, width, detail['layers'], batch_first=True,
                                  dropout=detail['dropout'])
        elif family == 'tcn':
            self.projection = nn.Conv1d(channels, width, 1)
            self.encoder = nn.Sequential(*(ResidualBlock(width, detail['kernel'], d, detail['dropout'])
                                           for d in detail['dilations']))
        else:
            steps = detail['input_steps']
            self.patch_starts = detail.get('patch_starts')
            self.patch_size = detail.get('patch_size', 1)
            self.projection = nn.Linear(channels * self.patch_size, width)
            self.positions = nn.Parameter(torch.empty(1, detail.get('patch_count', steps), width))
            nn.init.normal_(self.positions, std=.02)
            block = nn.TransformerEncoderLayer(width, detail['heads'], detail['feed_forward'], detail['dropout'],
                                               activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(block, detail['layers'], norm=nn.LayerNorm(width),
                                                 enable_nested_tensor=False)
        self.group_sizes = [len(TARGET_GROUPS[g]) for g in OUTPUT_GROUPS[detail['output']]]
        self.heads = nn.ModuleList(nn.Linear(width, self.horizon_count * size) for size in self.group_sizes)

    def forward(self, features):
        if self.family in ('gru', 'gru16'):
            _, hidden = self.encoder(features)
            last = hidden[-1]
        elif self.family == 'tcn':
            last = self.encoder(self.projection(features.transpose(1, 2)))[:, :, -1]
        else:
            if self.patch_starts is not None:
                features = torch.stack([features[:, start:start + self.patch_size].flatten(1)
                                        for start in self.patch_starts], dim=1)
            last = self.encoder(self.projection(features) + self.positions)[:, -1]
        return torch.cat([head(last).reshape(len(features), self.horizon_count, size)
                          for head, size in zip(self.heads, self.group_sizes)], dim=-1)
