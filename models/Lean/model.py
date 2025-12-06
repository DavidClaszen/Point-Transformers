import torch
import torch.nn as nn
from pointnet_util import farthest_point_sample, index_points, square_distance


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        # x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c
        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class StackedAttention(nn.Module):
    def __init__(self, channels=256, num_stacks=4, num_conv_layers=2):
        super().__init__()
        
        self.channels = channels
        self.num_conv_layers = num_conv_layers
        self.num_stacks = num_stacks
        
        if num_conv_layers >= 1:
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm1d(channels)
        if num_conv_layers >= 2:
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm1d(channels)
        
        # To make it more explicit for the ablation study, we define each SA layer separately
        if num_stacks >= 1:
            self.sa1 = SA_Layer(channels)
        if num_stacks >= 2:
            self.sa2 = SA_Layer(channels)
        if num_stacks >= 3:
            self.sa3 = SA_Layer(channels)
        if num_stacks >= 4:
            self.sa4 = SA_Layer(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        if hasattr(self, 'conv1'):
            x = self.relu(self.bn1(self.conv1(x)))
        if hasattr(self, 'conv2'):
            x = self.relu(self.bn2(self.conv2(x)))

        if hasattr(self, 'sa1'):
            x1 = self.sa1(x)
            x = x1
        if hasattr(self, 'sa2'):
            x2 = self.sa2(x1)
            x = x2
        if hasattr(self, 'sa3'):
            x3 = self.sa3(x2)
            x = x3
        if hasattr(self, 'sa4'):
            x4 = self.sa4(x3)
            x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.num_class
        d_points = cfg.input_dim
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=cfg.sa.channels)
        
        self.pt_last = StackedAttention(cfg.sa.channels, 
                                        num_stacks=cfg.sa.num_stacks, 
                                        num_conv_layers=cfg.sa.num_conv_layers)

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(cfg.sa.channels * (cfg.sa.num_stacks + 1), cfg.lbr_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(cfg.lbr_channels),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.num_lbrd = cfg.num_lbrd

        if cfg.decoder.num_lbrd == 1:
            self.linear1 = nn.Linear(cfg.lbr.channels, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p=0.5)
            self.linear3 = nn.Linear(512, output_channels)
        elif cfg.num_lbrd == 2:
            self.linear1 = nn.Linear(cfg.lbr.channels, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p=0.5)
            self.linear2 = nn.Linear(512, 256)
            self.bn7 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(p=0.5)
            self.linear3 = nn.Linear(256, output_channels)
        else:
            raise ValueError("cfg.decoder.num_lbrd must only be 1 or 2")

    def forward(self, x):
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        if self.num_lbrd == 1:
            x = self.relu(self.bn6(self.linear1(x)))
            x = self.dp1(x)
            x = self.linear3(x)
        elif self.num_lbrd == 2:
            x = self.relu(self.bn6(self.linear1(x)))
            x = self.dp1(x)
            x = self.relu(self.bn7(self.linear2(x)))
            x = self.dp2(x)
            x = self.linear3(x)

        return x