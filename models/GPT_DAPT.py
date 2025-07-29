import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 out_dim=None,
                 bottleneck=None,
                 dropout=0.0,
                 adapter_layernorm_option="in",
                 use_square=False, ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.use_square = use_square
        # _before
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
        self.scale = nn.Linear(self.n_embd, 1)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.GELU()
        if out_dim is None:
            self.up_proj = nn.Linear(self.down_size, self.n_embd)
        else:
            self.up_proj = nn.Linear(self.down_size, out_dim)

        self.dropout = dropout

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
            nn.init.zeros_(self.scale.bias)
            nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
            nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        scale = F.relu(self.scale(x))
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * scale
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output


def init_tfts(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta


def apply_tfts(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(hidden_features)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = apply_tfts(x, self.tfts_gamma_1, self.tfts_beta_1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = apply_tfts(x, self.tfts_gamma_2, self.tfts_beta_2)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.adapt_mlp = Adapter(d_model=embed_dim, bottleneck=64, dropout=0.)
        self.non_linear_func = nn.GELU()

        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(embed_dim)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(embed_dim)
        self.tfts_gamma_3, self.tfts_beta_3 = init_tfts(embed_dim)
        self.tfts_gamma_4, self.tfts_beta_4 = init_tfts(embed_dim * 4)
        self.tfts_gamma_5, self.tfts_beta_5 = init_tfts(embed_dim)

    def forward(self, x, attn_mask):
        x = apply_tfts(self.ln_1(x), self.tfts_gamma_1, self.tfts_beta_1)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.ln_2(x)
        m = apply_tfts(m, self.tfts_gamma_2, self.tfts_beta_2)
        a_mlp = self.adapt_mlp(m)
        m = self.mlp[2](self.mlp[1](apply_tfts(self.mlp[0](x), self.tfts_gamma_4, self.tfts_beta_4)))
        # m = self.mlp(m)
        m = apply_tfts(m, self.tfts_gamma_5, self.tfts_beta_5) + a_mlp
        # m = m + a_mlp
        x = x + m

        adapter_prompt = self.non_linear_func(a_mlp).mean(dim=0, keepdim=True)
        adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma_3, self.tfts_beta_3)
        x = torch.cat([x[0:2, :], adapter_prompt, x[2:, :]], dim=0)

        return x


class GPT_extractor(nn.Module):
    def __init__(
            self, embed_dim, num_heads, num_layers, num_classes, trans_dim, group_size, pretrained=False
    ):
        super(GPT_extractor, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * (self.group_size), 1)
        )

        if pretrained == False:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        self.cls_norm = nn.LayerNorm(self.trans_dim)
        self.num_group = 128

    def forward(self, h, pos, classify=False):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, length, C = h.shape

        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=h.device) * self.sos
        if not classify:
            h = torch.cat([sos, h[:-1, :, :]], axis=0)
        else:
            h = torch.cat([sos, h], axis=0)

        # attn_mask = torch.full(
        #     (length + 1, length + 1), -float("Inf"), device=h.device, dtype=h.dtype
        # ).to(torch.bool)
        # attn_mask = torch.triu(attn_mask, diagonal=1)

        # transformer
        # h = h + pos
        for i, layer in enumerate(self.layers):
            attn_mask = torch.full(
                (length + i + 1, length + i + 1), -float("Inf"), device=h.device, dtype=h.dtype
            ).to(torch.bool)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            h[-length - 1:, :] = pos + h[-length - 1:, :]
            h = layer(h, attn_mask)

        h = self.ln_f(h)

        encoded_points = h.transpose(0, 1)
        if not classify:
            return encoded_points

        h = h.transpose(0, 1)
        h = self.cls_norm(h)
        concat_f = torch.cat([h[:, 1], h[:, 2:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret, concat_f


class GPT_generator(nn.Module):
    def __init__(
            self, embed_dim, num_heads, num_layers, trans_dim, group_size
    ):
        super(GPT_generator, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * (self.group_size), 1)
        )

    def forward(self, h, pos, attn_mask):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, length, C = h.shape

        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)

        # transformer
        for layer in self.layers:
            h = layer(h + pos, attn_mask)

        h = self.ln_f(h)

        rebuild_points = self.increase_dim(h.transpose(1, 2)).transpose(
            1, 2).transpose(0, 1).reshape(batch * length, -1, 3)

        return rebuild_points
