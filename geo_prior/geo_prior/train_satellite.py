from comet_ml import Experiment

experiment = Experiment(api_key="IwUoq6LzlxwDePJx1egspBLcQ",
                         project_name="butterflai", workspace="martaskrt")
import magicpath
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import math
import os
import torch

import satellite_model as models
import geo_utils as ut
import datasets as dt
import grid_predictor as grid
from paths import get_paths
import losses_satellite as lo

from geo_utilities import ObjFromDict
import json
import argparse
assert magicpath
import pandas as pd
from data_loader import setup_dataset

from tqdm import tqdm
from calendar import monthrange
import calendar
import cv2
import tilenet

class LocationDataLoader(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, num_classes, is_train):
        self.labels = labels
        self.is_train = is_train
        self.num_classes = num_classes
        self.image_dir="static_maps_imgs_3kmx3km"
        self.imgs = imgs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        loc_class = self.labels[index]
        img_path = os.path.join(self.image_dir, self.imgs[index])
        img = cv2.imread(img_path)[:,:,::-1]
        #img = img.repeat(1, 1, 2)
        img = np.transpose(img, (2, 0, 1)) / 255
        img = torch.from_numpy(img).float().to(self.device)
        img = img.repeat(2,1,1)
        img = img[:4, :,:]
        
        return img, loc_class

def train(args, model, tile2vec_model, data_loader, optimizer, epoch, params):
    model.train()

    # adjust the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = params['lr'] * (params['lr_decay'] ** epoch)

    loss_avg = ut.AverageMeter()
    inds = torch.arange(params['batch_size']).to(params['device'])
    for batch_idx, (imgs, loc_class) in enumerate(data_loader):
        optimizer.zero_grad()
        loss = lo.embedding_loss(args, model, tile2vec_model, params, imgs, loc_class, inds)
        loss.backward()
        optimizer.step()

        loss_avg.update(loss.item(), len(loc_class))

        if (batch_idx % params['log_frequency'] == 0 and batch_idx != 0) or (batch_idx == (len(data_loader)-1)):
            print('[{}/{}]\tLoss  : {:.4f}'.format(batch_idx * params['batch_size'], len(data_loader.dataset), loss_avg.avg))
    experiment.log_metric("train_loss", loss_avg.avg, step=epoch)

def test(args, model, tile2vec_model, data_loader, params, epoch, split, val_loss_array=None, found=False, optimizer=None, early_stop_epoch=-1):
    # NOTE the test loss only tracks the log loss it is not the full loss used during training
    model.eval()
    loss_avg = ut.AverageMeter()

    inds = torch.arange(params['batch_size']).to(params['device'])
    with torch.no_grad():

        for imgs, loc_class in data_loader:
            tile2vec_emb = tile2vec_model.encode(imgs)
            loc_emb = model(tile2vec_emb)
            loc_pred = torch.sigmoid(model.class_emb(loc_emb))
            pos_loss = lo.log_loss(loc_pred[inds[:loc_class.shape[0]], loc_class])
            loss = pos_loss.mean()

            loss_avg.update(loss.item(), loc_class.shape[0])
    print('{} loss   : {:.4f}'.format(split, loss_avg.avg))
    if split == "val":
        experiment.log_metric("val_loss", loss_avg.avg, step=epoch)
        val_loss_array.append(loss_avg.avg)
        if val_loss_array[-1] == min(val_loss_array):
            print('Saving output model to ' + params['model_file_name'] + "_bestvalloss.pth.tar")
            op_state = {'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'params' : params}
            torch.save(op_state, params['model_file_name'] + "_bestvalloss.pth.tar")

        if args.early_stop_patience > -1:
            patience = args.early_stop_patience
            if not found and val_loss_array[-1] == min(val_loss_array):
    #        # save checkpoint
                print('Saving output model to ' + params['model_file_name'] + "_earlystop.pth.tar")
                op_state = {'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'params' : params}
                torch.save(op_state, params['model_file_name'] + "_earlystop.pth.tar")

            if not found and len(val_loss_array) > patience and val_loss_array[-patience] == min(val_loss_array):
                found=True
                early_stop_epoch = epoch-patience+1
                print("EARLY STOP EPOCH:::{}".format(epoch-patience+1))

        return val_loss_array, found, early_stop_epoch



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    parser.add_argument("--output")
    parser.add_argument("--elev", action="store_true")
    parser.add_argument("--date", action="store_true")
    parser.add_argument("--clean_data", action="store_true")
    parser.add_argument("--train_full", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=-1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--world_coords", action="store_true")
    parser.add_argument("--path_to_geo_bins", default=None)
    parser.add_argument("--train_ebd", action="store_true")
    parser.add_argument("--use_imgs", action="store_true")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--pos_weight", default=10, type=int)
    parser.add_argument("--prop_pos_weight", action="store_true")
    parser.add_argument("--ebd_pretrain", default=None, help="path to ebd model for weight init")
    parser.add_argument("--ebd_fc", default=None, help="path to ebd model to build fc layer from")
    args = parser.parse_args()
    experiment.set_name(args.output)
    with open(args.config_file) as json_file:
        config_dict = json.load(json_file)
    config_dict['clean_data'] = args.clean_data
    #config_dict['geo_bins'] = False
    config_dict['geo_bins'] = args.path_to_geo_bins
    #config_dict['geo'] = '/home/zach/marta/InsectClassification/utils/bin_data_n20m20.pickle'
    config = ObjFromDict(config_dict)
    # hyper params
    params = {}
    params['dataset'] = 'eButterfly'  # inat_2018, inat_2017, birdsnap, nabirds, yfcc
    params['batch_size'] = 32
    params['lr'] = 0.0005
    params['lr_decay'] = 0.98
    params['num_filts'] = 256  # embedding dimension
    params['num_epochs'] = args.epochs
    params['log_frequency'] = 50
    params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    params['balanced_train_loader'] = True
    params['max_num_exs_per_class'] = 100
    params['map_range'] = (-180, 180, -90, 90)

    # specify feature encoding for location and date
    params['use_date_feats'] = args.date  # if False date feature is not used
    params['loc_encode']     = 'encode_cos_sin'  # encode_cos_sin, encode_3D, encode_none
    params['date_encode']    = 'encode_cos_sin' # encode_cos_sin, encode_none

    # specify loss type
    # appending '_user' models the user location and object affinity - see losses.py
    params['train_loss'] = 'full_loss'  # full_loss_user, full_loss

    print('Dataset   \t' + params['dataset'])
    print(config.clean_data)
    # op = dt.load_dataset(params, 'val', True, True)

    train_data, val_data, num_classes = setup_dataset(config, debug=False, return_dataloader=False)

    train_data.samples = train_data.samples[train_data.samples.Longitude.notnull()]
    train_data.samples = train_data.samples[train_data.samples.Latitude.notnull()]
    
    train_imgs = train_data.samples['Satellite_imgs'].values
    val_imgs = val_data.samples['Satellite_imgs'].values
    train_classes = train_data.samples['label'].values
    val_classes = val_data.samples['label'].values

    if args.train_full:
        train_data_path, val_data_path, test_data_path = train_data.samples['path'].values, val_data.samples['path'].values, test_data.samples['path'].values
        # df = pd.read_csv("all_records_with_photo_2020_06_18_06_51_29_STANDARDIZED.csv")
        df = pd.read_csv("all_records_w_imgs.csv")
        if not args.world_coords:
            df = df[(df.Latitude >= 5) & (df.Latitude <= 75)]
            df = df[(df.Longitude >= -175) & (df.Longitude <= -25)]
        species = df[["Genus", "Species"]].values
        satellite_imgs = df['Satellite_imgs'].values
        classes = ["{} {}".format(item[0], item[1]) for item in species]

        full_train_classes, full_val_classes, full_test_classes, extra_classes = [], [], [], []
        full_train_imgs, full_val_imgs, full_test_imgs, extra_imgs = [],[],[],[]

        le = train_data.label_encoder
        for i in tqdm(range(len(path))):
            if classes[i] not in le.classes_:
                continue
            if int(locs[i][0]) > 500 or int(locs[i][1]) > 500:
                continue
            if path[i] in train_data_path:
                full_train_classes.append(classes[i]); full_train_imgs.append(satellite_imgs[i])
            elif path[i] in val_data_path:
                full_val_classes.append(classes[i]); full_val_imgs.append(satellite_imgs[i])
            else:
                n, p = 1, .9
                s = np.random.binomial(n, p, 1)
                if s[0] == 0:
                    full_val_classes.append(classes[i]); full_val_imgs.append(dates[i])
                elif s[0] == 1:
                    full_train_classes.append(classes[i]); full_train_imgs.append(dates[i])

        train_classes, val_classes, test_classes = np.array(le.transform(full_train_classes)), np.array(le.transform(full_val_classes)), np.array(le.transform(full_test_classes))
        train_imgs, val_imgs, test_imgs = np.array(full_train_imgs), np.array(full_val_imgs), np.array(full_test_imgs)

    params['num_classes'] = num_classes
    params['meta_type'] = ''
    if args.output:
        params['model_file_name'] = args.output
    op_dir = 'ims/ims_' + params['dataset'] + '/'
    if not os.path.isdir(op_dir):
        os.makedirs(op_dir)

    # process users
    # NOTE we are only modelling the users in the train set - not the val
    #un_users, train_users = np.unique(train_users, return_inverse=True)
    #train_users = torch.from_numpy(train_users).to(params['device'])
    params['num_users'] = 0
    #if 'user' in params['train_loss']:
     #   assert (params['num_users'] != 1)  # need to have more than one user

    satellite_img_dir = "static_maps_imgs_3kmx3km/"
    print(train_classes[:10])
    print(train_imgs[:10])
    train_classes = [train_classes[i] for i in range(len(train_classes)) if os.path.exists(satellite_img_dir + str(train_imgs[i]))]
    train_imgs = [img for img in train_imgs if os.path.exists(satellite_img_dir +  str(img))]
    assert len(train_classes) == len(train_imgs)
    val_classes = [val_classes[i] for i in range(len(val_classes)) if os.path.exists(satellite_img_dir + str(val_imgs[i]))]
    val_imgs = [img for img in val_imgs if os.path.exists(satellite_img_dir + str(img))]
    assert len(val_classes) == len(val_imgs)
    # print stats
    print('\nnum_classes\t{}'.format(params['num_classes']))
    print('num train    \t{}'.format(len(train_imgs)))

    print('num val      \t{}'.format(len(val_imgs)))

    # data loaders
    train_classes = np.array(train_classes)
    train_labels = torch.from_numpy(train_classes).to(params['device'])
    train_dataset = LocationDataLoader(train_imgs, train_labels,params['num_classes'], True)

    if params['balanced_train_loader']:
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=params['batch_size'],
                       sampler=ut.BalancedSampler(train_classes.tolist(), params['max_num_exs_per_class'],
                       use_replace=False, multi_label=False), shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=params['batch_size'], shuffle=True)
    val_classes = np.array(val_classes)
    val_labels = torch.from_numpy(val_classes).to(params['device'])
    val_dataset = LocationDataLoader(val_imgs, val_labels, params['num_classes'], False)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=params['batch_size'], shuffle=False)


    # create model
    #params['num_feats'] = train_classes.shape[1]
    model = models.Satellite_Model(num_classes=params['num_classes'], in_channels=18432).to(params['device'])

    
    
    tile2vec_model = tilenet.make_tilenet().to(params['device'])
    training_state = torch.load('/home/zach/marta/tile2vec/models/naip_trained.ckpt')
    state_dict = tile2vec_model.state_dict()
    for k in training_state:
        if k in state_dict:
            state_dict[k] = training_state[k]
    tile2vec_model.load_state_dict(state_dict)
    tile2vec_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # main train loop
    val_loss_array = []
    found=False
    early_stop_epoch=-1
    for epoch in tqdm(range(0, params['num_epochs'])):
        print('\nEpoch\t{}'.format(epoch))
        train(args, model, tile2vec_model, train_loader, optimizer, epoch, params)
        val_loss_array, found, early_stop_epoch = test(args, model, tile2vec_model, val_loader, params, epoch,
                split="val", val_loss_array=val_loss_array, found=found, optimizer=optimizer, early_stop_epoch=early_stop_epoch)

        print(val_loss_array, found)
    print("EARLY_STOP_EPOCH:::{}".format(early_stop_epoch))
    print("BEST_VAL_LOSS_EPOCH:::{}".format(val_loss_array.index(min(val_loss_array))))


    # save trained model
    print('Saving output model to ' + params['model_file_name'] + "_lastepoch.pth.tar")
    op_state = {'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'params' : params}
    torch.save(op_state, params['model_file_name'] + "_lastepoch.pth.tar")

if __name__== "__main__":
    main()
