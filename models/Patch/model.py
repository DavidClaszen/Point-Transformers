import torch
import torch.nn as nn
import math
from pointnet_util import farthest_point_sample, index_points, square_distance


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)      # [B, npoint, N]
    _, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)

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
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
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


class SA_Layer_RP(nn.Module):
    """
    Copy of SA_Layer, with relative positional bias added
    """
    def __init__(self, channels, pos_hidden_dim=16):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # tiny MLP: R^3 -> R^1 for relative positional bias
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_hidden_dim, 1)
        )

    def forward(self, x, xyz):
        """
        x:   (B, C, N)   features
        xyz: (B, N, 3)   coordinates for these N points
        """
        x_q = self.q_conv(x).permute(0, 2, 1)  # (B, N, Cq)
        x_k = self.k_conv(x)                   # (B, Cq, N)
        x_v = self.v_conv(x)                   # (B, C,  N)
        energy = torch.bmm(x_q, x_k)           # (B, N, N)

        # bias
        rel = xyz[:, :, None, :] - xyz[:, None, :, :]
        pos_bias = self.pos_mlp(rel).squeeze(-1)

        # add bias to energy
        energy = energy + pos_bias
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class SA_Layer_MH(nn.Module):
    """
    NOT USED. DIDN'T RESULT IN PERFORMANCE GAIN
    Multi-head version of your original SA_Layer.

    - Same offset attention formulation:
        energy = q @ k
        attention = softmax(energy)
        attention = attention / (sum over query dim)

    - q and k share the same conv (like q_conv.weight = k_conv.weight).
    - No sqrt(d_k) scaling.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim_v = channels // num_heads  # for v

        # q/k use a reduced dimension like original: channels // 4 total
        self.qk_channels = channels // 4
        assert self.qk_channels % num_heads == 0, "channels//4 must be divisible by num_heads"
        self.head_dim_qk = self.qk_channels // num_heads
        self.qk_conv = nn.Conv1d(channels, self.qk_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)

        # offset + residual
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (B, C, N)
        returns: (B, C, N)
        """
        B, C, N = x.shape

        # qk: (B, C_qk, N)
        qk = self.qk_conv(x)
        # v:  (B, C, N)
        v = self.v_conv(x)

        # reshape into heads
        # qk: (B, H, Dh_qk, N)
        qk = qk.view(B, self.num_heads, self.head_dim_qk, N)
        # v:  (B, H, Dh_v,  N)
        v = v.view(B, self.num_heads, self.head_dim_v, N)

        # q: (B, H, N, Dh_qk), k: (B, H, Dh_qk, N)
        q = qk.permute(0, 1, 3, 2)
        k = qk
        # energy: (B, H, N, N)
        energy = torch.matmul(q, k)
        attention = self.softmax(energy)

        # Weirdly functional...
        # extra renormalization over query dimension (dim=2 in (B,H,N,N)):
        # makes columns roughly sum to 1 as well (doubly-stochastic-ish)
        attention = attention / (1e-9 + attention.sum(dim=2, keepdim=True))
        x_r = torch.matmul(v, attention)

        # merge heads back: (B, C, N)
        x_r = x_r.view(B, C, N)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class StackedAttention(nn.Module):
    def __init__(self, channels=256, pos_hidden_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer_RP(channels, pos_hidden_dim=pos_hidden_dim)
        self.sa2 = SA_Layer_RP(channels, pos_hidden_dim=pos_hidden_dim)
        self.sa3 = SA_Layer_RP(channels, pos_hidden_dim=pos_hidden_dim)
        self.sa4 = SA_Layer_RP(channels, pos_hidden_dim=pos_hidden_dim)

        self.relu = nn.ReLU()

    def forward(self, x, xyz):
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x,  xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)

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
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention(channels=256, pos_hidden_dim=16)

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

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
        new_xyz, new_feature = sample_and_group(
            npoint=256, nsample=32, xyz=new_xyz, points=feature
        )
        feature_1 = self.gather_local_1(new_feature)      # (B, C, 256)

        x = self.pt_last(feature_1, new_xyz)              # new_xyz: (B, 256, 3)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x
