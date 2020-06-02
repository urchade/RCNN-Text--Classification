import torch
from torch import nn


class RCNN(nn.Module):
    def __init__(self, n_emb, emb_dim, c_size, h_size, out_size=1, pad_idx=0):
        super(RCNN, self).__init__()

        self.embedding = nn.Embedding(n_emb, emb_dim, pad_idx)

        self.W_l = nn.Parameter(torch.randn(size=(c_size, c_size)))
        self.W_r = nn.Parameter(torch.randn(size=(c_size, c_size)))
        self.W_sl = nn.Parameter(torch.randn(size=(emb_dim, c_size)))
        self.W_sr = nn.Parameter(torch.randn(size=(emb_dim, c_size)))

        self.linear_2 = nn.Linear(emb_dim + 2 * c_size, h_size)
        self.linear_out = nn.Linear(h_size, out_size)

        self._init = nn.Parameter(torch.zeros(size=(2, c_size)), requires_grad=False)

    def forward(self, seq_ids):
        out_emb = self.embedding(seq_ids)  # (batch_size, seq_len, emb_dim)

        _, seq_len, _ = out_emb.size()

        cl_wi, cr_wi = self._init

        cl, cr = [], []
        for i in range(seq_len):
            cl_wi = torch.tanh(cl_wi @ self.W_l + out_emb[:, i, :] @ self.W_sl)
            cr_wi = torch.tanh(cr_wi @ self.W_r + out_emb[:, seq_len - 1 - i, :] @ self.W_sr)
            cl.append(cl_wi)
            cr.append(cr_wi)

        cl = torch.stack(cl).transpose(0, 1)  # (batch_size, seq_len, c_size)
        cr = torch.stack(cr).transpose(0, 1)  # (batch_size, seq_len, c_size)

        x = torch.cat([cl, out_emb, cr], dim=2)  # (batch_size, seq_len, emb_dim + 2 * c_size)
        y_2 = torch.tanh(self.linear_2(x))  # (batch_size, seq_len, h_size)
        y_3 = y_2.max(dim=1)[0]  # (batch_size, h_size)

        y_4 = self.linear_out(y_3)  # (batch_size, out_size)

        return y_4
