"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import os
import torch
import time
from model.model import BQModel
from utils.chekpointer import CheckPointer
from utils.misc import set_seed, print_model_params_info, maybe_cuda_ddp


def setup_exp(args, problem="tsp", is_test=False):
    print(args)
    set_seed(args.seed)

    print("Using", torch.cuda.device_count(), "GPU(s)")

    for d in [args.output_dir, os.path.join(args.output_dir, "models")]:
        if not os.path.exists(d):
            os.makedirs(d)

    if problem == "tsp":
        node_input_dim = 2
    elif problem == "kp":
        node_input_dim = 3
    elif problem == "cvrp" or problem == "op":
        node_input_dim = 4

    net = BQModel(node_input_dim, args.dim_emb, args.dim_ff, args.activation_ff, args.nb_layers_encoder,
                  args.nb_heads, args.activation_attention, args.dropout, args.batchnorm, problem)

    net, module, device = maybe_cuda_ddp(net)
    print_model_params_info(module)

    optimizer = None
    if not is_test:
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr) if not args.test_only else None

    if args.pretrained_model != "":
        path = args.pretrained_model
        model_dir, name = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
        checkpointer = CheckPointer(name=name, save_dir=model_dir)
        _, other = checkpointer.load(module, optimizer, label='best', map_location=device)
    else:
        model_dir, name = os.path.join(args.output_dir, "models"), f'{int(time.time() * 1000.0)}'
        checkpointer = CheckPointer(name=name, save_dir=model_dir)
        other = None

    return net, module, device, optimizer, checkpointer, other
