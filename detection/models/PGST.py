import torch
import torch.nn as nn
from timm.models.layers import DropPath
from knn_cuda import KNN
from utils import misc

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


def get_laplacian(adj_matrix, normalize=True):
    """
    Compute the graph Laplacian matrix.

    Args:
        adj_matrix (torch.Tensor): The adjacency matrix (batch_size, vertices, vertices).
        normalize (bool): Whether to compute the normalized Laplacian.

    Returns:
        torch.Tensor: The Laplacian matrix (batch_size, vertices, vertices).
    """
    if normalize:
        # Degree matrix: sum of rows
        D = torch.sum(adj_matrix, dim=-1)  # (batch_size, vertices)
        # Avoid division by zero by adding epsilon to D
        D_inv_sqrt = torch.rsqrt(D + 1e-6)  # Inverse square root
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)  # Batch-wise diagonal matrices
        # Normalized Laplacian
        L = torch.eye(adj_matrix.size(-1), device=adj_matrix.device) - \
            D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    else:
        # Degree matrix
        D = torch.sum(adj_matrix, dim=-1)  # (batch_size, vertices)
        D = torch.diag_embed(D)  # Batch-wise diagonal matrices
        # Unnormalized Laplacian
        L = D - adj_matrix
    return L

def get_basis(center):
    L = torch.cdist(center, center)
    L = 1 / (L / torch.min(L[L > 0], dim=-1, keepdim=True).values + torch.eye(L.size(-1), device=L.device).unsqueeze(0))
    L = get_laplacian(L)
    _, U = torch.linalg.eigh(L)
    return U # This should be "U.transpose(-2, -1)", we keep it for reproducing our results in paper.

def sort(pts: torch.Tensor, idx: torch.Tensor):
    return torch.gather(pts, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, pts.size(-1)))

class PCSA(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = rank
        self.norm_ly1 = nn.LayerNorm(self.rank)
        self.norm_ly2 = nn.LayerNorm(self.rank)

        self.act = nn.SiLU()
        self.down = nn.Linear(dim, self.rank)
        self.up = nn.Linear(self.rank, dim)

        self.scale=1.

        self.adapt1 = nn.Linear(self.rank, self.rank)
        nn.init.zeros_(self.adapt1.weight)
        nn.init.zeros_(self.adapt1.bias)
        
        self.drop_adapt1 = DropPath(0.)
        self.drop_adapt2 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, input, sub_U, idx):
        h = self.down(input)
        h = self.act(h)
        x = h

        B0, group_num, group_size, _ = sub_U[0].shape
        G0 = group_num * group_size
        sub_x0=sort(x,idx[0])
        sub_x0 = sub_x0.reshape(B0, group_num, group_size, self.rank)

        B1, group_num, group_size, _ = sub_U[1].shape
        sub_x1 = sub_x0.reshape(B1, group_num, group_size, self.rank)

        sub_x_f0 = sub_U[0].transpose(-2, -1) @ sub_x0
        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt2(self.act(self.drop_out(self.adapt1(sub_x_f0))))
        sub_x0 = sub_U[0] @ sub_x_f0

        sub_x0 = sub_x0.reshape(B0, G0, self.rank)
        sub_x0=sort(sub_x0,idx[1])

        sub_x_f1 = sub_U[1].transpose(-2, -1) @ sub_x1
        sub_h_f1 = sub_x_f1
        sub_x_f1 = self.norm_ly2(sub_x_f1)
        sub_x_f1 = sub_h_f1 + self.drop_adapt2(self.act(self.drop_out(self.adapt1(sub_x_f1))))
        sub_x1 = sub_U[1] @ sub_x_f1

        sub_x1 = sub_x1.reshape(B0, G0, self.rank)
        sub_x0=sort(sub_x1,idx[1])

        x = sub_x0 + sub_x1

        h = x + h
        h = self.up(h)
        return h
