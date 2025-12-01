# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from utils.misc import summary_parameters


def build_optimizer(args, model):
    param_groups = add_weight_decay(model, args, weight_decay=0.05, optimizer_part=args.optimizer_part)
    summary_parameters(model)
    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


def add_weight_decay(model, args, weight_decay=1e-5, skip_list=(), optimizer_part='only_new'):
    print('only the following parameter is optimized')
    decay = []
    no_decay = []
    num_trainable_params = 0
    for name, param in model.named_parameters():
        if optimizer_part == 'only_new':
            if ('mlp_heads' in name) or ('decoder' in name) or ('pre_encoder' in name) or (
                    'query' in name) or ('pos' in name) or ('encoder' in name):
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                    num_trainable_params += param.numel()
                else:
                    decay.append(param)
                    num_trainable_params += param.numel()
                print(name)
            else:
                param.requires_grad = False
        elif optimizer_part == 'adapt':
            if ('mlp_heads' in name) or ('decoder' in name) or ('pre_encoder' in name) or (
                    'query' in name) or ('adapt' in name) or ('pos' in name) or ('cls' in name) or ('Adapter'in name):
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    no_decay.append(param)
                    num_trainable_params += param.numel()
                else:
                    decay.append(param)
                    num_trainable_params += param.numel()
                print(name)
            else:
                param.requires_grad = False
        elif optimizer_part == 'all':
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                # print(name)
                no_decay.append(param)
                num_trainable_params += param.numel()
            else:
                decay.append(param)
                num_trainable_params += param.numel()
            print(name)

    return [
        {'params': no_decay, 'weight_decay': 0., 'lr': args.base_lr},
        {'params': decay, 'weight_decay': weight_decay, 'lr': args.base_lr},
    ]
