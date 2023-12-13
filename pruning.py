import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__(in_features, out_features, bias=bias, device=device)
        self.register_buffer('mask', torch.ones(out_features, in_features, dtype=torch.bool, device=device))
        self.n_pruned = 0
        self.rows_pruned = torch.zeros(in_features, dtype=torch.bool, device=device)

    def forward(self, input):
        with torch.no_grad():
            weight = self.weight * self.mask
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


class CompressedLinear(nn.Linear):
    def __init__(self, orig_in_features, compressed_in_features, out_features, rows_not_pruned, bias=True, device=None):
        super().__init__(compressed_in_features, out_features, bias=bias, device=device)
        self.orig_in_features = orig_in_features
        self.register_buffer('rows_not_pruned', rows_not_pruned)

    def forward(self, input):
        super().forward(input * self.rows_not_pruned)

def convert_to_prunable(model, device=None):
    names, modules = [], []
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            convert_to_prunable(module, device)  # compound module: recurse
        if isinstance(module, nn.Linear):
            names.append(name)
            modules.append(module)

    # Replace Linear versions with PrunableLinear
    for name, module in zip(names, modules):
        prunable = PrunableLinear(module.in_features, module.out_features, bias=(module.bias is not None), device=device)
        prunable.weight = module.weight
        prunable.bias = module.bias
        setattr(model, name, prunable)

def compress_layers(model, device=None):
    names, modules = [], []
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            compress_layers(module, device)  # compound module: recurse
        if isinstance(module, PrunableLinear):
            names.append(name)
            modules.append(module)

    # Replace PrunableLinear modules with CompressedLinear versions
    for name, module in zip(names, modules):
        setattr(model, name, compress(module, device))
    
    return sum(p.numel() for p in model.parameters())


def compress(module: PrunableLinear, device=None) -> 'CompressedLinear':
    rows_not_pruned = ~module.rows_pruned
    compressed_in_features = torch.sum(rows_not_pruned).item()
    compressed = CompressedLinear(module.in_features, compressed_in_features, module.out_features, rows_not_pruned, bias=(module.bias is not None), device=device)
    del compressed.weight
    compressed.weight = nn.Parameter(module.weight[:, rows_not_pruned])
    if module.bias is not None:
        del compressed.bias
        compressed.bias = nn.Parameter(module.bias)
    return compressed
