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


def get_land_coords(args, params):
    r_lon = np.random.randint(-175, -25, 1)[0]
    r_lat = np.random.randint(5, 75, 1)[0]
    date = np.random.uniform(-1, 1)
    while not globe.is_land(r_lat, r_lon):
        r_lon = np.random.randint(-175, -25, 1)[0]
        r_lat = np.random.randint(5, 75, 1)[0]

    rand_feats_orig = torch.FloatTensor([r_lon/180, r_lat/90, date]).unsqueeze(0).to('cuda')
    return ut.encode_loc_time(args, rand_feats_orig[:,:2], rand_feats_orig[:,2], concat_dim=1, params=params)

def rand_samples(args, batch_size, params, rand_type='uniform', labels=None, label2feats=None):

    # randomly sample background locations
    if args.elev:
        rand_feats_orig = torch.rand(batch_size, 2).to(params['device'])*2 -1
        rand_feats_elev = torch.rand(batch_size, 1).to(params['device'])
        rand_feats_orig = torch.cat((rand_feats_orig, rand_feats_elev), 1)
        rand_feats_date = torch.rand(batch_size, 1).to(params['device'])*2 -1
        rand_feats_orig = torch.cat((rand_feats_orig, rand_feats_date), 1)
    else:
        rand_feats_orig = torch.rand(batch_size, 3).to(params['device'])*2 -1
    if rand_type == 'spherical':
        theta = ((rand_feats_orig[:,1].unsqueeze(1)+1) / 2.0)*(2*math.pi)
        r_lon = torch.sqrt(1.0 - rand_feats_orig[:,0].unsqueeze(1)**2) * torch.cos(theta)
        r_lat = torch.sqrt(1.0 - rand_feats_orig[:,0].unsqueeze(1)**2) * torch.sin(theta)
        if args.elev:
            rand_feats_orig = torch.cat((r_lon, r_lat, rand_feats_orig[:,2:]), 1)
        else:
            rand_feats_orig = torch.cat((r_lon, r_lat, rand_feats_orig[:,2].unsqueeze(1)), 1)

    if args.elev:
        rand_feats = ut.encode_loc_time(args, rand_feats_orig[:,:3], rand_feats_orig[:,3], concat_dim=1, params=params)
    else:
        rand_feats = ut.encode_loc_time(args, rand_feats_orig[:,:2], rand_feats_orig[:,2], concat_dim=1, params=params)
    
    return rand_feats


def embedding_loss(args, model, params, loc_feat, loc_class, user_ids, inds, ebd=False, imgs=None, label2feats=None):

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # create random background samples
    loc_feat_rand = rand_samples(args, batch_size, params, rand_type='spherical', labels=loc_class, label2feats=label2feats)
    # get location embeddings
    loc_emb_rand = model(loc_feat_rand, return_feats=True)
    loc_emb = model(loc_feat, imgs=imgs, return_feats=True)
    if ebd:
        loc_pred = torch.sigmoid(model.class_emb(model.ebd_emb(loc_emb)))
        loc_pred_rand = torch.sigmoid(model.class_emb(model.ebd_emb(loc_emb_rand)))
    else:
        loc_pred = torch.sigmoid(model.class_emb(loc_emb))
        loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))
   # pos_weight = params['num_classes']
    if args.prop_pos_weight:
        pos_weight = []
        total_samples = 0
        for label in label2feats:
            total_samples += len(label2feats[label])
        for class_ in loc_class:
            pos_weight.append(total_samples/len(label2feats[class_.item()]))
        pos_weight = torch.FloatTensor(pos_weight).to(device)
    else:
        pos_weight = args.pos_weight
    loss_pos = log_loss(1.0 - loc_pred)  # neg
     
    loss_pos[inds[:batch_size], loc_class] = pos_weight*log_loss(loc_pred[inds[:batch_size], loc_class])  # pos
    loss_bg = log_loss(1.0 - loc_pred_rand)
    if 'user' in params['train_loss']:

        # user location loss
        user = model.user_emb.weight[user_ids, :]
        p_u_given_l = torch.sigmoid((user*loc_emb).sum(1))
        p_u_given_randl = torch.sigmoid((user*loc_emb_rand).sum(1))

        user_loc_pos_loss = log_loss(p_u_given_l)
        user_loc_neg_loss = log_loss(1.0 - p_u_given_randl)

        # user class loss
        p_c_given_u = torch.sigmoid(torch.matmul(user, model.class_emb.weight.transpose(0,1)))
        user_class_loss = log_loss(1.0 - p_c_given_u)
        user_class_loss[inds[:batch_size], loc_class] = pos_weight*log_loss(p_c_given_u[inds[:batch_size], loc_class])

        # total loss
        loss = loss_pos.mean() + loss_bg.mean() + user_loc_pos_loss.mean() + \
               user_loc_neg_loss.mean() + user_class_loss.mean()

    else:
        #pass
        # total loss
        loss = loss_pos.mean() + loss_bg.mean()
    return loss
    #return loc_pred, loc_pred_rand_pos, loc_pred_rand_neg
