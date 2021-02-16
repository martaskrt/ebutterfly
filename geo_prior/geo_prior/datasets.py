import os
import numpy as np
import json
import pandas as pd
from scipy import sparse
import sys
sys.path.append('../')
from paths import get_paths


def load_dataset(params, eval_split, train_remove_invalid, eval_remove_invalid,
                 load_cnn_predictions=False, load_cnn_features=False,
                 load_cnn_features_train=False):

    if params['dataset'] == 'inat_2017':

        data_dir = get_paths('inat_2017_data_dir')
        num_classes = 5089
        class_of_interest = 3731

        # load observations
        train_locs, train_classes, train_users, train_dates, train_inds = \
            load_inat_data(data_dir, 'train2017_locations.json',
            'train2017.json', train_remove_invalid)
        if eval_split == 'val':
            val_locs, val_classes, val_users, val_dates, val_inds = \
                load_inat_data(data_dir, eval_split+'2017_locations.json',
                eval_split+'2017.json', eval_remove_invalid)
        elif eval_split == 'test':
            val_locs, val_classes, val_users, val_dates, val_inds = \
                load_inat_data(data_dir, eval_split+'2017_locations.json',
                eval_split+'2017_DO_NOT_SHARE.json', eval_remove_invalid)
            val_split = pd.read_csv(data_dir + 'kaggle_solution_2017_DO_NOT_SHARE.csv')['usage'].values == 'Private'

        # load class names
        with open(data_dir + 'categories2017.json') as da:
            cls_data = json.load(da)
        class_names = [cc['name'] for cc in cls_data]
        class_ids = [cc['id'] for cc in cls_data]
        classes = dict(zip(class_ids, class_names))

        if load_cnn_predictions:
            val_preds = load_sparse_feats(data_dir + 'features_inception/inat2017_' + eval_split + '_preds_sparse.npz')

        if load_cnn_features:
            val_feats = np.load(data_dir + 'features_inception/inat2017_' + eval_split + '_net_feats.npy')

        if load_cnn_features_train:
            train_feats = np.load(data_dir + 'features_inception/inat2017_train_net_feats.npy')


    elif params['dataset'] == 'inat_2018':

        data_dir = get_paths('inat_2018_data_dir')
        num_classes = 8142
        class_of_interest = 3731  # wood thrush

        # load observations
        train_locs, train_classes, train_users, train_dates, train_inds = \
            load_inat_data(data_dir, 'train2018_locations.json',
            'train2018.json', train_remove_invalid)
        if eval_split == 'val':
            val_locs, val_classes, val_users, val_dates, val_inds = \
                load_inat_data(data_dir, eval_split+'2018_locations.json',
                eval_split+'2018.json', eval_remove_invalid)
        elif eval_split == 'test':
            val_locs, val_classes, val_users, val_dates, val_inds = \
                load_inat_data(data_dir, eval_split+'2018_locations.json',
                eval_split+'2018_DO_NOT_SHARE.json', eval_remove_invalid)
            val_split = pd.read_csv(data_dir + 'kaggle_solution_2018_DO_NOT_SHARE.csv')['usage'].values == 'Private'

        # load class names
        with open(data_dir + 'categories2018.json') as da:
            cls_data = json.load(da)
        class_names = [cc['name'] for cc in cls_data]
        class_ids = [cc['id'] for cc in cls_data]
        classes = dict(zip(class_ids, class_names))

        if load_cnn_predictions:
            if params['inat2018_resolution'] == 'high_res':
                val_preds = load_sparse_feats(data_dir + 'features_inception_hr/inat2018_' + eval_split + '_preds_sparse.npz')
            else:
                val_preds = load_sparse_feats(data_dir + 'features_inception/inat2018_' + eval_split + '_preds_sparse.npz')

        if load_cnn_features:
            if params['inat2018_resolution'] == 'high_res':
                val_feats = np.load(data_dir + 'features_inception_hr/inat2018_' + eval_split + '_net_feats.npy')
            else:
                val_feats = np.load(data_dir + 'features_inception/inat2018_' + eval_split + '_net_feats.npy')

        if load_cnn_features_train:
            if params['inat2018_resolution'] == 'high_res':
                train_feats = np.load(data_dir + 'features_inception_hr/inat2018_train_net_feats.npy')
            else:
                train_feats = np.load(data_dir + 'features_inception/inat2018_train_net_feats.npy')


    elif params['dataset'] == 'birdsnap':

        data_dir = get_paths('birdsnap_data_dir')
        ann_file_name = 'birdsnap_with_loc_2019.json'
        num_classes = 500
        class_of_interest = 0

        # load observations
        train_locs, train_classes, train_users, train_dates, train_inds = \
            load_bird_data(data_dir, ann_file_name, 'train', train_remove_invalid, params['meta_type'])
        val_locs, val_classes, val_users, val_dates, val_inds = \
            load_bird_data(data_dir, ann_file_name, eval_split, eval_remove_invalid, params['meta_type'])

        # load class names
        with open(data_dir + ann_file_name) as da:
            class_names = json.load(da)['classes']
        classes = dict(zip(range(len(class_names)), class_names))

        if load_cnn_predictions:
            val_preds = load_sparse_feats(data_dir + 'features_inception/birdsnap_' + eval_split + '_preds_sparse.npz')

        if load_cnn_features:
            val_feats = np.load(data_dir + 'features_inception/birdsnap_' + eval_split + '_net_feats.npy')

        if load_cnn_features_train:
            train_feats = np.load(data_dir + 'features_inception/birdsnap_train_net_feats.npy')


    elif params['dataset'] == 'nabirds':

        data_dir = get_paths('nabirds_data_dir')
        ann_file_name = 'nabirds_with_loc_2019.json'
        num_classes = 555
        class_of_interest = 0

        # load observations
        train_locs, train_classes, train_users, train_dates, train_inds = \
            load_bird_data(data_dir, ann_file_name, 'train', train_remove_invalid, params['meta_type'])
        val_locs, val_classes, val_users, val_dates, val_inds = \
            load_bird_data(data_dir, ann_file_name, eval_split, eval_remove_invalid, params['meta_type'])

        # load class names
        with open(data_dir + ann_file_name) as da:
            class_names = json.load(da)['classes']
        classes = dict(zip(range(len(class_names)), class_names))

        if load_cnn_predictions:
            val_preds = load_sparse_feats(data_dir + 'features_inception/nabirds_' + eval_split + '_preds_sparse.npz')

        if load_cnn_features:
            val_feats = np.load(data_dir + 'features_inception/nabirds_' + eval_split + '_net_feats.npy')

        if load_cnn_features_train:
            train_feats = np.load(data_dir + 'features_inception/nabirds_train_net_feats.npy')


    elif params['dataset'] == 'yfcc':

        data_dir = get_paths('yfcc_data_dir')
        print('  No user or date features for yfcc.')
        params['use_date_feats'] = False
        params['balanced_train_loader'] = False
        num_classes = 100
        class_of_interest = 9  # beach

        # load observations
        train_locs, train_classes, train_users, train_dates = load_yfcc_data(data_dir, 'train_test_split.csv', 'train')
        val_locs, val_classes, val_users, val_dates = load_yfcc_data(data_dir, 'train_test_split.csv', eval_split)
        train_inds = np.arange(train_locs.shape[0])
        val_inds = np.arange(val_locs.shape[0])

        # load class names
        da = pd.read_csv(data_dir + 'class_names.csv')
        classes = dict(zip(da['id'].values, da['name'].values))

        if load_cnn_predictions:
            val_preds = np.load(data_dir + 'features_inception/YFCC_' + eval_split + '_preds.npy')

        if load_cnn_features:
            val_feats = np.load(data_dir + 'features_inception/YFCC_' + eval_split + '_net_feats.npy')

        if load_cnn_features_train:
            train_feats = np.load(data_dir + 'features_inception/YFCC_train_net_feats.npy')


    if load_cnn_features_train and train_remove_invalid:
        train_feats = train_feats[train_inds, :]

    if load_cnn_features and eval_remove_invalid:
        val_feats = val_feats[val_inds, :]
        val_preds = val_preds[val_inds, :]


    # return data in dictionary
    op = {}
    op['train_locs'] = train_locs
    op['train_classes'] = train_classes
    op['train_users'] = train_users
    op['train_dates'] = train_dates

    op['val_locs'] = val_locs
    op['val_classes'] = val_classes
    op['val_users'] = val_users
    op['val_dates'] = val_dates

    op['class_of_interest'] = class_of_interest
    op['classes'] = classes
    op['num_classes'] = num_classes

    if load_cnn_predictions:
        op['val_preds'] = val_preds  # class predictions from trained image classifier
    if load_cnn_features:
        op['val_feats'] = val_feats  # features from trained image classifier
        assert val_feats.shape[0] == val_locs.shape[0]
    if load_cnn_features_train:
        op['train_feats'] = train_feats  # features from trained image classifier
        assert train_feats.shape[0] == train_locs.shape[0]

    # if it exists add the data split
    try:
        op['val_split'] = val_split
    except:
        op['val_split'] = np.ones(val_locs.shape[0], dtype=np.int)

    return op


def load_sparse_feats(file_path, invert=False):
    feats = sparse.load_npz(file_path)
    feats = np.array(feats.todense(), dtype=np.float32)
    if invert:
        eps = 10e-5
        feats = np.clip(feats, eps, 1.0-eps)
        feats = np.log(feats/(1.0-feats))
    return feats


def load_bird_data(ip_dir, ann_file_name, split_name, remove_empty=False, meta_type='orig_meta'):

    print('Loading ' + os.path.basename(ann_file_name) + ' - ' + split_name)
    print('   using meta data: ' + meta_type)

    # load annotation info
    with open(ip_dir + ann_file_name) as da:
        data_orig = json.load(da)

    data = [dd for dd in data_orig[split_name] if dd['valid_image']]
    imgs = np.array([dd['im_path'] for dd in data])
    classes = np.array([dd['class_id'] for dd in data]).astype(np.int)
    users = [dd[meta_type]['user_id'] for dd in data]
    users = np.array([-1 if uu is None else uu for uu in users]).astype(np.int)
    dates = np.array([dd[meta_type]['date'] for dd in data]).astype(np.float32)
    lon = [dd[meta_type]['lon'] for dd in data]
    lat = [dd[meta_type]['lat'] for dd in data]
    locs = (np.vstack((lon, lat)).T).astype(np.float32)

    print('\t {} total entries'.format(len(data_orig[split_name])))
    print('\t {} entries with images'.format(len(data)))

    valid_inds = (~np.isnan(locs[:, 0])) & (users >=0) & (~np.isnan(dates))
    if remove_empty:
        locs = locs[valid_inds, :]
        users = users[valid_inds]
        dates = dates[valid_inds]
        classes = classes[valid_inds]

    print('\t {} entries with meta data'.format(valid_inds.sum()))
    if not remove_empty:
        print('\t keeping entries even without metadata')

    return locs, classes, users, dates, valid_inds


def load_inat_data(ip_dir, loc_file_name, ann_file_name, remove_empty=False):
    # TODO clean this up and remove loop
    print('Loading ' + os.path.basename(loc_file_name))

    # load location info
    with open(ip_dir + loc_file_name) as da:
        loc_data = json.load(da)
    loc_data_dict = dict(zip([ll['id'] for ll in loc_data], loc_data))

    if '_large' in loc_file_name:
        # special case where the loc data also includes meta data such as class
        locs = [[ll['lon'], ll['lat']] for ll in loc_data]
        dates = [ll['date_c'] for ll in loc_data]
        classes = [ll['class'] for ll in loc_data]
        users = [ll['user_id'] for ll in loc_data]
        keep_inds = np.arange(len(locs))
        print('\t {} valid entries'.format(len(locs)))

    else:
        # otherwise load regualar iNat data

        # load annotation info
        with open(ip_dir + ann_file_name) as da:
            data = json.load(da)

        ids = [tt['id'] for tt in data['images']]
        ids_all = [ii['image_id'] for ii in data['annotations']]
        classes_all = [ii['category_id'] for ii in data['annotations']]
        classes_mapping = dict(zip(ids_all, classes_all))

        # store locations and associated classes
        locs    = []
        classes = []
        users   = []
        dates   = []
        miss_cnt = 0
        keep_inds = []
        for ii, tt in enumerate(ids):

            if remove_empty and ((loc_data_dict[tt]['lon'] is None) or (loc_data_dict[tt]['user_id'] is None)):
                miss_cnt += 1
            else:
                if (loc_data_dict[tt]['lon'] is None):
                    loc = [np.nan, np.nan]
                else:
                    loc = [loc_data_dict[tt]['lon'], loc_data_dict[tt]['lat']]

                if (loc_data_dict[tt]['user_id'] is None):
                    u_id = -1
                else:
                    u_id = loc_data_dict[tt]['user_id']

                locs.append(loc)
                classes.append(classes_mapping[int(tt)])
                users.append(u_id)
                dates.append(loc_data_dict[tt]['date_c'])
                keep_inds.append(ii)

        print('\t {} valid entries'.format(len(locs)))
        if remove_empty:
            print('\t {} entries excluded with missing meta data'.format(miss_cnt))

    return np.array(locs).astype(np.float32), np.array(classes).astype(np.int), \
           np.array(users).astype(np.int), np.array(dates).astype(np.float32), np.array(keep_inds)


def load_yfcc_data(data_dir, ann_file_name, split_name):
    da = pd.read_csv(data_dir + ann_file_name)
    locs = da[da['split'] == split_name][['lon', 'lat']].values.astype(np.float32)
    classes = da[da['split'] == split_name]['class'].values
    users = np.ones(locs.shape[0], dtype=np.int)*-1
    dates = np.zeros(locs.shape[0], dtype=np.float32)
    return locs, classes, users, dates
