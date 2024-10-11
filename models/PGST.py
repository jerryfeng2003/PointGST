import torch
import torch.nn as nn
from timm.models.layers import DropPath


class GetLaplacian(nn.Module):
    def __init__(self, normalize=True):
        super(GetLaplacian, self).__init__()
        self.normalize = normalize

    def diag(self, mat):
        # input is batch x vertices
        d = []
        for vec in mat:
            d.append(torch.diag(vec))
        return torch.stack(d)

    def forward(self, adj_matrix):
        if self.normalize:
            D = torch.sum(adj_matrix, dim=-1)
            eye = torch.ones_like(D)
            eye = self.diag(eye)
            D = 1 / torch.sqrt(D)
            D = self.diag(D)
            L = eye - D * adj_matrix * D
        else:
            D = torch.sum(adj_matrix, dim=-1)
            D = self.diag(D)
            L = D - adj_matrix
        return L

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


def sort(pts:torch.Tensor,idx:torch.Tensor):
    batch_indices = torch.arange(pts.size(0)).unsqueeze(1).expand_as(idx)
    sorted_pts = pts[batch_indices, idx]
    return sorted_pts