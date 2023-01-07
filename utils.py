import random
import numpy as np
import torch


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])

    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    category_targets = [f["category_target"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    category_targets = torch.tensor(category_targets, dtype=torch.float)

    output = (input_ids, input_mask, category_targets)
    return output


def collate_fn_v2(batch):
    # hierarchical
    max_len = max([len(f["input_ids"]) for f in batch])

    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    entity_targets = [f["entity_target"] for f in batch]
    aspect_targets = [f["aspect_target"] for f in batch]
    category_targets = [f["category_target"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    entity_targets = torch.tensor(entity_targets, dtype=torch.float)
    aspect_targets = torch.tensor(aspect_targets, dtype=torch.float)
    category_targets = torch.tensor(category_targets, dtype=torch.float)

    output = (input_ids, input_mask, entity_targets, aspect_targets, category_targets)
    return output