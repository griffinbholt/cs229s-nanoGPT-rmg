import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features, dtype=torch.bool))
        self.n_pruned = 0
        self.rows_pruned = torch.zeros(in_features, dtype=torch.bool)

    def forward(self, input):
        return F.linear(input, self.weight*self.mask, self.bias)

    def prune(self, p: float = None, m: float = None, rowwise: bool = False) -> int:
        """
        p: percentile to prune (if using percentage-based pruning)
        m: magnitude below which to prune (if using magnitude-based pruning)
        rowwise: prune entire rows, if True (accomplished by looking at
                 magnitude / percentile of norm of row, normalized by size of
                 the row – so that smaller dimension rows aren't pruned more often)

        Returns the number of parameters that were pruned
        """
        assert (p is not None) or (m is not None)  # one of them is available
        assert (p is None) or (m is None)          # can't be both at the same time
        if p is not None:  # only prunes weights not already pruned (for pcnt)
            if rowwise:
                m = torch.quantile(torch.norm(self.weight[self.mask], dim=0) / self.weight.shape[0])
            else:
                m = torch.quantile(torch.abs(self.weight[self.mask]), p)
        if rowwise:
            pruned = torch.where((torch.norm(self.weight, dim=0) / self.weight.shape[0] < m))[0]
            self.mask[:, pruned] = 0
            self.rows_pruned[pruned] = True
            n_pruned = (len(pruned) * self.out_features) - self.n_pruned
        else:
            pruned = torch.where(torch.abs(self.weight) < m)
            self.mask[pruned] = 0
            n_pruned = len(pruned[0]) - self.n_pruned
        self.n_pruned += n_pruned
        return n_pruned

    def compress(self) -> 'CompressedLinear':
        rows_not_pruned = ~self.rows_pruned
        compressed_in_features = torch.sum(rows_not_pruned).item()
        compressed = CompressedLinear(self.in_features, compressed_in_features, self.out_features, rows_not_pruned, bias=(self.bias is not None))
        compressed.weight = nn.Parameter(self.weight[:, rows_not_pruned])
        compressed.bias = nn.Parameter(self.bias)
        return compressed


class CompressedLinear(nn.Linear):
    def __init__(self, orig_in_features, compressed_in_features, out_features, rows_not_pruned, bias=True):
        super().__init__(compressed_in_features, out_features, bias)
        self.orig_in_features = orig_in_features
        self.rows_not_pruned = rows_not_pruned

    def forward(self, input):
        if input.shape[-1] == self.orig_in_features:
            return super().forward(input[:, self.rows_not_pruned])
        return super().forward(input)


def compress_layers(model):
    names, modules = [], []
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            compress_layers(module)  # compound module: recurse
        if isinstance(module, PrunableLinear):
            names.append(name)
            modules.append(module)

    # Replace PrunableLinear modules with CompressedLinear versions
    for name, module in zip(names, modules):
        setattr(model, name, module.compress())
    
    return sum(p.numel() for p in model.parameters())


# class MInner(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1 = PrunableLinear(3, 10)
#         self.l2 = PrunableLinear(10, 3)

#     def forward(self, x):
#         return self.l2(self.l2(x))


# class MOuter(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.M1 = MInner()
#         self.M2 = MInner()
#         self.n_pruned = 0
#         self.n_parameters = sum(p.numel() for p in self.parameters())

#     def forward(self, x):
#         return self.M1(self.M2(x))

#     def prune(self, p: float = None, m: float = None, rowwise: bool = False) -> float:
#         total_pruned = 0
#         for module in self.modules():
#             if isinstance(module, PrunableLinear):
#                 total_pruned += module.prune(p, m, rowwise)
#         self.n_pruned += total_pruned
#         pcnt_pruned = self.n_pruned / self.n_parameters
#         return pcnt_pruned # Total Percentage Pruned

