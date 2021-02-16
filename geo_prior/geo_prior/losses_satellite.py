import torch
import geo_utils as ut
import math
import pickle
import numpy as np
np.random.seed(42)
from global_land_mask import globe
from torch import nn


def log_loss(pred):
    return -torch.log(pred + 1e-5)




def embedding_loss(args, model, tile2vec_model, params, imgs, loc_class, inds):

    assert model.inc_bias == False
    batch_size = imgs.shape[0]
    with torch.no_grad():
        tile2vec_emb = tile2vec_model.encode(imgs)
    loc_emb = model(tile2vec_emb)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    pos_weight = args.pos_weight
    loss_pos = log_loss(1.0 - loc_pred)  # neg
     
    loss_pos[inds[:batch_size], loc_class] = pos_weight*log_loss(loc_pred[inds[:batch_size], loc_class])  # pos
    loss = loss_pos.mean()
    return loss
