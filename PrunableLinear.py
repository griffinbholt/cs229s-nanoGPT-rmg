import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features, dtype=torch.bool))
        self.n_pruned = 0

    def forward(self, input):
        return F.linear(input, self.weight*self.mask, self.bias)

    def prune(self, p: float = None, m: float = None, rowwise: bool = False) -> int:
        """
        p: percentile to prune (if using percentage-based pruning)
        m: magnitude below which to prune (if using magnitude-based pruning)
        rowwise: if using magnitude-pruning, then it will prune entire rows

        Returns the number of parameters that were pruned
        """
        assert (p is not None) or (m is not None)  # one of them is available
        assert (p is None) or (m is None)          # can't be both at the same time
        if p is not None:
            assert not rowwise  # can't do rowwise and percentage based pruning
            with torch.no_grad():
                m = torch.quantile(torch.abs(self.weight[self.mask]), p)  # only prunes weights not already pruned
        if rowwise:
            pruned = torch.where((torch.norm(self.weight, dim=0) < m))[0]
            self.mask[:, pruned] = 0
            n_pruned = (len(pruned) * self.out_features) - self.n_pruned
        else:
            pruned = torch.where(torch.abs(self.weight) < m)
            self.mask[pruned] = 0
            n_pruned = len(pruned[0]) - self.n_pruned
        self.n_pruned += n_pruned
        return n_pruned

# class MInner(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1 = PrunableLinear(3, 10)
#         self.l2 = PrunableLinear(10, 3)

#     def forward(self, x):
#         return self.l2(self.l2(x))


# class MOuter(nn.Module):
#     def __init__(self):
    #     super().__init__()
    #     self.M1 = MInner()
    #     self.M2 = MInner()
    #     self.n_pruned = 0
    #     self.n_parameters = sum(p.numel() for p in self.parameters())
        
    # def forward(self, x):
    #     return self.M1(self.M2(x))

    # def prune(self, p: float = None, m: float = None, rowwise: bool = False) -> float:
    #     total_pruned = 0
    #     for module in self.modules():
    #         if isinstance(module, PrunableLinear):
    #             total_pruned += module.prune(p, m, rowwise)
    #     self.n_pruned += total_pruned
    #     pcnt_pruned = self.n_pruned / self.n_parameters
    #     return pcnt_pruned # Total Percentage Pruned