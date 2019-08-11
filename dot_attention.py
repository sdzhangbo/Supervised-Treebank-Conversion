import torch
import torch.nn as nn
import torch.nn.functional as F

class DotAttention(nn.Module):

    def __init__(self, guide_dim, hidden_dim, batch_first=False):
        super(DotAttention, self).__init__()
        self._hidden_dim = hidden_dim
        self._guide_dim = guide_dim
        self.align = nn.Linear(guide_dim, hidden_dim)
        self.batch_first = batch_first

    def forward(self, guides, hiddens):
        assert len(guides.size()) >= 2
        if not self.batch_first:
            guides, hiddens = guides.transpose(0, 1), hiddens.transpose(0, 1)

        q, k, v = self.align(guides), hiddens.transpose(-2, -1), hiddens
        result = torch.bmm(F.softmax(torch.bmm(q, k), dim=1), v)
        if not self.batch_first:
            result = result.transpose(0, 1)
        return result

        #return torch.bmm(F.softmax(torch.bmm(q, k), dim=1), v).unbind(dim=0)

class PerceptronAttention(nn.Module):

    def __init__(self, guide_dim, hidden_dim):
        super(PerceptronAttention, self).__init__()
        self._hidden_dim = hidden_dim
        self._guide_dim = guide_dim
        self.align = nn.Linear(guide_dim, hidden_dim)


if __name__ == '__main__':
    attn = DotAttention(20, 10)
    guide = torch.randn(5, 2, 20)
    hiddens = torch.randn(5, 2, 10)
    result = attn(guide, hiddens)
    print(result)
    print(result.size())
