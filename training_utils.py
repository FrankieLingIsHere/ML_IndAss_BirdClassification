import torch
import re

def freeze_batchnorm_stats(module):
    """Put all BatchNorm layers in eval mode and freeze their parameters.

    This prevents BatchNorm from updating running_mean/var when batch sizes are small.
    """
    for m in module.modules():
        # support both nn.BatchNorm2d and other BN variants
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def unfreeze_batchnorm_stats(module):
    """Restore BatchNorm layers to training mode (allows running stats update) and enable grad for their weights/biases."""
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.train()
            for p in m.parameters():
                p.requires_grad = True


def gradual_unfreeze(model, backbone_attr='backbone', block_name_pattern=None, unfreeze_last_n_blocks=2):
    """Unfreeze the last N blocks of a backbone (best-effort).

    - model: model object (expects attribute backbone_attr to exist)
    - block_name_pattern: regex to identify block names (e.g. r'_blocks\\.\\d+' for EfficientNet)
    - unfreeze_last_n_blocks: integer number of blocks to unfreeze from the end

    Returns list of parameter names that were unfrozen.
    """
    backbone = getattr(model, backbone_attr, None)
    if backbone is None:
        return []

    # Collect named modules/parameters and try to detect block indices
    named = list(backbone.named_children())
    unfrozen = []

    # If a block pattern provided, use named_modules to detect blocks
    if block_name_pattern:
        pattern = re.compile(block_name_pattern)
        block_names = [name for name, _ in backbone.named_modules() if pattern.search(name)]
        # Deduplicate and sort
        block_names = sorted(list(dict.fromkeys(block_names)))
        to_unfreeze = block_names[-unfreeze_last_n_blocks:]
        for bn in to_unfreeze:
            sub = dict(backbone.named_modules()).get(bn, None)
            if sub is None:
                continue
            for p_name, p in sub.named_parameters():
                full = f"{bn}.{p_name}"
                p.requires_grad = True
                unfrozen.append(full)
    else:
        # Fallback: unfreeze last N child modules
        child_names = [n for n, _ in named]
        to_unfreeze = child_names[-unfreeze_last_n_blocks:]
        for cn in to_unfreeze:
            sub = dict(backbone.named_children()).get(cn, None)
            if sub is None:
                continue
            for p_name, p in sub.named_parameters():
                full = f"{cn}.{p_name}"
                p.requires_grad = True
                unfrozen.append(full)

    return unfrozen
