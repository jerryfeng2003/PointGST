import torch
import torch.nn as nn
from timm.models.layers import DropPath


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
    def __init__(self, dim, cfg):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = cfg.rank
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

    def forward(self, input, U, sub_U, idx):
        h = self.down(input)

        B0, group_num, group_size, _ = sub_U.shape
        G0 = group_num * group_size

        x = h[:, -G0:, :]
        h = self.act(h)
        
        sub_x0=sort(x,idx[0])
        sub_x0 = sub_x0.reshape(B0, group_num, group_size, self.rank)
        
        x_f = U @ x
        h_f = x_f
        x_f = self.norm_ly1(x_f)
        x_f = h_f + self.drop_adapt1(self.act(self.drop_out(self.adapt1(x_f))))
        x = U.transpose(-2, -1) @ x_f
        
        sub_x_f0 = sub_U @ sub_x0
        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt2(self.act(self.drop_out(self.adapt1(sub_x_f0))))
        sub_x0 = sub_U.transpose(-2, -1) @ sub_x_f0

        sub_x0 = sub_x0.reshape(B0, G0, self.rank)
        sub_x0=sort(sub_x0,idx[1])

        x = x + sub_x0

        h[:, -G0:, :] = x + h[:, -G0:, :]
        h = self.up(h)
        return h*self.scale
