import os
import json

import torch

from utils.utilities import ObjFromDict
from models.resnet import Resnet
from models.wide_resnet import WideResnet
from models.residual_attention_network import ResidualAttentionModel_448input


def load_state_dict(run_dir, metric):
    checkpoint_path = os.path.join(run_dir, 'best_{}.pth'.format(metric))
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    return state_dict


def load_model(run_dir, metric):
    with open(os.path.join(run_dir, 'config.json')) as json_file:
        config = json.load(json_file)
    config['model']['finetune_from'] = False
    config = ObjFromDict(config)
    model = get_model(config.model)
    state_dict = load_state_dict(run_dir, metric)
    model.load_state_dict(state_dict)
    return model, config


def get_model(config, loss_type=None):
    if config.name == 'resnet':
        model = Resnet(config, loss_type)

    elif config.name == 'wide_resnet':
        model = WideResnet(config, loss_type)

    elif config.name == 'attention_resnet':
        model = ResidualAttentionModel_448input(config)

    if config.finetune_from:
        model.cpu()
        model_dict = model.state_dict()
        # get the pretrained state dict and remove the unnecessary key
        pretrained_state_dict = load_state_dict(config.finetune_from,
                                                metric=config.finetune_from_best)
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items()
                                 if 'backbone'in k}
        model_dict.update(pretrained_state_dict)
        model.load_state_dict(model_dict)
    return model
