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

import geo_models as models
import geo_utils as ut
import datasets as dt
import grid_predictor as grid
from paths import get_paths
import losses as lo

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


def transform_date(date):
    try:
        year, month, day = date.split("-")
        year, month, day = int(year), int(month)-1, int(day)-1
        month_count = np.cumsum([monthrange(year, ii+1)[1] for ii in range(12)])
        month_count = np.hstack(([0], month_count))
        num_days = 365.0 if calendar.isleap(year) else 364.0
        dt = (month_count[month] + day) / 365.0
        dt = round(dt,6)
    except:
        dt = np.nan
    return dt

class LocationDataLoader(torch.utils.data.Dataset):
    def __init__(self, loc_feats, labels, users, num_classes, is_train, imgs=None):
        self.loc_feats = loc_feats
        self.labels = labels
        self.users = users
        self.is_train = is_train
        self.num_classes = num_classes
        self.image_dir="static_maps_imgs"
        self.imgs = imgs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def __len__(self):
        return len(self.loc_feats)

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc_class = self.labels[index]
        if self.users:
            users = self.users[index]
            if self.is_train:
                return loc_feat, loc_class, user
        elif len(self.imgs) > 0:
            img_path = os.path.join(self.image_dir, self.imgs[index])
            img = cv2.imread(img_path)[:,:,::-1]
            img = np.transpose(img, (2, 0, 1)) / 255
            img = torch.from_numpy(img).float().to(self.device)
            return loc_feat, loc_class, img
        else:
            return loc_feat, loc_class

def generate_feats(args, locs, dates, params):
    x_locs = ut.convert_loc_to_tensor(args, locs, params['device'])
    if params['use_date_feats']:
        x_dates = torch.from_numpy(dates.astype(np.float32)*2 - 1).to(params['device'])
    else:
        x_dates = None
    feats = ut.encode_loc_time(args, x_locs, x_dates, concat_dim=1, params=params)
    return feats


def train(args, model, data_loader, optimizer, epoch, params, label2feats=None):
    model.train()

    # adjust the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = params['lr'] * (params['lr_decay'] ** epoch)

    loss_avg = ut.AverageMeter()
    inds = torch.arange(params['batch_size']).to(params['device'])
    for batch_idx, (loc_feat, loc_class) in enumerate(data_loader):
        optimizer.zero_grad()
        user_ids = None
        ebd=False
        imgs=[]
        if args.ebd_fc:
            ebd=True
        loss = lo.embedding_loss(args, model, params, loc_feat, loc_class, user_ids, inds, ebd=ebd, imgs=imgs, label2feats=label2feats)
        loss.backward()
        optimizer.step()

        loss_avg.update(loss.item(), len(loc_feat))

        if (batch_idx % params['log_frequency'] == 0 and batch_idx != 0) or (batch_idx == (len(data_loader)-1)):
            print('[{}/{}]\tLoss  : {:.4f}'.format(batch_idx * params['batch_size'], len(data_loader.dataset), loss_avg.avg))
    experiment.log_metric("train_loss", loss_avg.avg, step=epoch)

def test(args, model, data_loader, params, epoch, split, val_loss_array=None, found=False, optimizer=None, early_stop_epoch=-1):
    # NOTE the test loss only tracks the log loss it is not the full loss used during training
    model.eval()
    loss_avg = ut.AverageMeter()

    inds = torch.arange(params['batch_size']).to(params['device'])
    with torch.no_grad():

        for loc_feat, loc_class in data_loader:
            imgs=[]
            loc_pred = model(loc_feat, imgs=imgs)
            pos_loss = lo.log_loss(loc_pred[inds[:loc_feat.shape[0]], loc_class])
            loss = pos_loss.mean()

            loss_avg.update(loss.item(), loc_feat.shape[0])
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


def plot_gt_locations(params, mask, train_classes, class_of_interest, classes, train_locs, train_dates, op_dir):
    # plot GT locations for the class of interest
    im_width  = (params['map_range'][1] - params['map_range'][0]) // 45  # 8
    im_height = (params['map_range'][3] - params['map_range'][2]) // 45  # 4
    plt.figure(num=0, figsize=[im_width, im_height])
    plt.imshow(mask, extent=params['map_range'], cmap='tab20')

    inds = np.where(train_classes==class_of_interest)[0]
    print('{} instances of: '.format(len(inds)) + classes[class_of_interest])

    # the color of the dot indicates the date
    colors = np.sin(np.pi*train_dates[inds])
    plt.scatter(train_locs[inds, 0], train_locs[inds, 1], c=colors, s=2, cmap='magma', vmin=0, vmax=1)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().set_frame_on(False)

    op_file_name = op_dir + 'gt_' + str(class_of_interest).zfill(4) + '.jpg'
    plt.savefig(op_file_name, dpi=400, bbox_inches='tight',pad_inches=0)


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
    config_dict['geo_bins'] = args.path_to_geo_bins
    config = ObjFromDict(config_dict)
    # hyper params
    params = {}
    params['dataset'] = 'eButterfly'  # inat_2018, inat_2017, birdsnap, nabirds, yfcc
    params['batch_size'] = 64
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

    if args.train_ebd:
        train_locs = train_data.samples[["Longitude", "Latitude"]].values.astype(np.float32)
        val_locs = val_data.samples[["Longitude", "Latitude"]].values.astype(np.float32)
    else:
        test_data = setup_dataset(config, debug=False, test_only=True, return_dataloader=False)
        train_data.samples = train_data.samples[train_data.samples.Longitude.notnull()]
        train_data.samples = train_data.samples[train_data.samples.Latitude.notnull()]
        
        train_locs = train_data.samples[["Longitude", "Latitude", "Elevation"]].values.astype(np.float32)
        val_locs = val_data.samples[["Longitude", "Latitude", "Elevation"]].values.astype(np.float32)
        test_locs = test_data.samples[["Longitude", "Latitude", "Elevation"]].values.astype(np.float32)
        test_classes = test_data.samples['label'].values
        test_dates = np.array(list(map(lambda x: transform_date(x), test_data.samples['Date Observed'].values)))
        test_users= None
    train_classes = train_data.samples['label'].values
    val_classes = val_data.samples['label'].values
    train_users = None
    train_dates = np.array(list(map(lambda x: transform_date(x), train_data.samples['Date Observed'].values)))
    val_users = None
    val_dates = np.array(list(map(lambda x: transform_date(x), val_data.samples['Date Observed'].values)))

    if args.train_full:
        print("TRAIN DATA:::{}".format(len(train_locs)))
        print("VAL DATA:::{}".format(len(val_locs)))
        train_data_path, val_data_path, test_data_path = train_data.samples['path'].values, val_data.samples['path'].values, test_data.samples['path'].values
        df = pd.read_csv("all_records_w_imgs.csv")
         
        if not args.world_coords:
            df = df[(df.Latitude >= 5) & (df.Latitude <= 75)]
            df = df[(df.Longitude >= -175) & (df.Longitude <= -25)]
        path = df['img_2'].values
        locs = df[["Longitude", "Latitude", "Elevation"]].values.astype(np.float32) # drop these if out of bounds
        species = df[["Genus", "Species"]].values
        classes = ["{} {}".format(item[0], item[1]) for item in species]
        dates = np.array(list(map(lambda x: transform_date(x), df['Date Observed'].values)))

        full_train_locs, full_val_locs, full_test_locs, extra_locs = [], [], [], []
        full_train_classes, full_val_classes, full_test_classes, extra_classes = [], [], [], []
        full_train_dates, full_val_dates, full_test_dates, extra_dates = [], [], [], []
        full_train_imgs, full_val_imgs, full_test_imgs, extra_imgs = [],[],[],[]\
        train_locs = list(train_locs); train_classes = list(train_classes); train_dates = list(train_dates)
        le = train_data.label_encoder
        orig_classes = []
        for i in tqdm(range(len(locs))):
            if classes[i] not in le.classes_:
                print(classes[i])
                continue
            if args.use_imgs:
                if satellite_imgs == "None":
                    continue
            if int(locs[i][0]) > 500 or int(locs[i][1]) > 500:
               continue
            if path[i] in train_data_path:
                full_train_locs.append(locs[i]); full_train_classes.append(classes[i]); full_train_dates.append(dates[i]); 
                full_train_imgs.append(satellite_imgs[i])
            elif path[i] in val_data_path:
                full_val_locs.append(locs[i]); full_val_classes.append(classes[i]); full_val_dates.append(dates[i]); 
                full_val_imgs.append(satellite_imgs[i])
            elif path[i] in test_data_path:
                full_test_locs.append(locs[i]); full_test_classes.append(classes[i]); full_test_dates.append(dates[i])
                full_test_imgs.append(satellite_imgs[i])
            else:
                n, p = 1, .9
                s = np.random.binomial(n, p, 1)
                if s[0] == 0:
                    full_val_locs.append(locs[i]); full_val_classes.append(classes[i]); full_val_dates.append(dates[i]); 
                    full_val_imgs.append(dates[i])
                elif s[0] == 1:
                    full_train_locs.append(locs[i]); full_train_classes.append(classes[i]); full_train_dates.append(dates[i]); 
                    full_train_imgs.append(dates[i])
        train_locs, val_locs, test_locs = np.array(full_train_locs), np.array(full_val_locs), np.array(full_test_locs)
        train_classes, val_classes, test_classes = np.array(le.transform(full_train_classes)), np.array(le.transform(full_val_classes)), np.array(le.transform(full_test_classes))
        train_dates, val_dates, test_dates = np.array(full_train_dates), np.array(full_val_dates), np.array(full_test_dates)
        train_imgs, val_imgs, test_imgs = np.array(full_train_imgs), np.array(full_val_imgs), np.array(full_test_imgs)

    print("TRAIN DATA:::{}".format(len(train_locs)))
    print("VAL DATA:::{}".format(len(val_locs)))
    if not args.train_ebd:
        print("TEST DATA:::{}".format(len(test_locs)))
    #class_of_interest = op['class_of_interest']
    #classes = op['classes']
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

    # print stats
    print('\nnum_classes\t{}'.format(params['num_classes']))
    print('num train    \t{}'.format(len(train_locs)))
    print('num val      \t{}'.format(len(val_locs)))
    print('train loss   \t' + params['train_loss'])
    print('model name   \t' + params['model_file_name'])
    # print('num users    \t{}'.format(params['num_users']))
    if params['meta_type'] != '':
        print('meta data    \t' + params['meta_type'])

    # load ocean mask for plotting
    mask = np.load(get_paths('mask_dir') + 'ocean_mask.npy').astype(np.int)

    # data loaders
    if not args.use_imgs:
        train_imgs, val_imgs, test_imgs = [], [], []
    train_labels = torch.from_numpy(train_classes).to(params['device'])
    train_feats = generate_feats(args, train_locs, dates=train_dates, params=params)

    label2feats = {}
    for i in range(len(train_labels)):
        if train_labels[i].item() not in label2feats:
            label2feats[train_labels[i].item()] = []
        label2feats[train_labels[i].item()].append(train_feats[i])

    if args.upsample:
        np.random.seed(42)
        train_labels_upsampled = []
        train_feats_upsampled = []
        train_classes = []
        for label in sorted(label2feats):
            num_samples = 100
            curr_train_feats = np.random.choice(label2feats[label], size=num_samples, replace=True)
            for row in curr_train_feats:
                train_feats_upsampled.append(row)
            train_labels_upsampled += [label] * num_samples
            train_classes += [label] * num_samples
        train_feats = torch.stack(train_feats_upsampled)
        train_labels = torch.tensor(train_labels_upsampled)
        train_classes = np.array(train_classes)
       # assert len(train_feats) == 601*num_samples and len(train_labels)== 601 *num_samples

    print(train_feats.shape, train_labels.shape)
    train_dataset = LocationDataLoader(train_feats, train_labels, train_users, params['num_classes'], True, imgs=train_imgs)
    if params['balanced_train_loader']:
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=params['batch_size'],
                       sampler=ut.BalancedSampler(train_classes.tolist(), params['max_num_exs_per_class'],
                       use_replace=False, multi_label=False), shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=params['batch_size'], shuffle=True)
    print(len(train_loader))
    val_labels = torch.from_numpy(val_classes).to(params['device'])
    val_feats = generate_feats(args, val_locs, dates=val_dates, params= params)
    val_dataset = LocationDataLoader(val_feats, val_labels, val_users, params['num_classes'], False, imgs=val_imgs)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=params['batch_size'], shuffle=False)

    if not args.train_ebd:
        test_labels = torch.from_numpy(test_classes).to(params['device'])
        test_feats = generate_feats(args, test_locs, dates=test_dates, params= params)

        test_dataset = LocationDataLoader(test_feats, test_labels, test_users, params['num_classes'], False, imgs=test_imgs)
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=params['batch_size'], shuffle=False)

    # create model
    params['num_feats'] = train_feats.shape[1]
    ebd=False
    if args.ebd_fc:
        ebd=True
    model = models.FCNet(num_inputs=params['num_feats'], num_classes=params['num_classes'],
                         num_filts=params['num_filts'], num_users=params['num_users'], ebd=ebd).to(params['device'])

    if args.ebd_pretrain or args.ebd_fc:
        checkpoint = args.ebd_pretrain or args.ebd_fc
        pretrained_dict = torch.load(checkpoint, map_location='cpu')['state_dict']
        model_dict = model.state_dict()
        if args.ebd_fc:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            pretrained_dict['ebd_emb.0.weight'] = pretrained_dict['class_emb.weight']
            pretrained_dict['class_emb.weight'] = model_dict['class_emb.weight']
        if args.ebd_pretrain:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('class_emb')}
            pretrained_dict['class_emb.weight'] = model_dict['class_emb.weight']
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        print("model loaded from checkpoint:::{}".format(checkpoint))
    if args.ebd_fc:
        for param in model.parameters():
            param.requires_grad = False
        model.class_emb.weight.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # main train loop
    val_loss_array = []
    found=False
    early_stop_epoch=-1
    for epoch in tqdm(range(0, params['num_epochs'])):
        print('\nEpoch\t{}'.format(epoch))
        train(args, model, train_loader, optimizer, epoch, params, label2feats)
        val_loss_array, found, early_stop_epoch = test(args, model, val_loader, params, epoch,
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
