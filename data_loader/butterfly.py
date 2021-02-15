import os

import numpy as np
import pandas as pd
import cv2
from sklearn import preprocessing as sklearn_preprocessing

from data_loader.preprocessing import Preprocessing

import pickle

class Encoder(sklearn_preprocessing.LabelEncoder):
    '''
    The Encoder instance inherits from the sklearn_preprocessing.LabelEncoder
    class, it aims to assign a one-hot vector to each label and keep track of
    the hierarchy (useful for adequate metric computation in the train file).

    Attributes
    ----------
    labels_to_parents : dict
        Dictionnary containing hierarchical information on the dataset's labels.

    family_to_species | genus_to_species : dict
        Dictionnary of the labels of interest grouped by higher label value in
        the hierarchical tree.
    '''

    def __init__(self):
        super().__init__()
        self.labels_to_parents = {}
        self.le = None
    def fit(self, dataset, fine_grained_level):
        '''
        Creates inherently the attribute labels_to_parents which contains the
        encoded labels grouped per hierarchical labels.

        Parameters
        ----------
        fine_grained_level : str
            The title of the label of interest  in the hierarchy.
            Either 'Family', 'Genus' or 'Species'.
        '''
        self.fine_grained_level = fine_grained_level
        self.le = super().fit(dataset[fine_grained_level])
        new_label = self.le.transform(dataset[fine_grained_level])
        encoded_labels = pd.DataFrame(new_label,
                                      columns=['label'],
                                      index=dataset.index)
        dataset = pd.concat([dataset, encoded_labels], axis=1)
        self.one_hot = sklearn_preprocessing.OneHotEncoder(sparse=False)
        reshaped_label = dataset['label'].values.reshape(-1, 1)
        self.one_hot.fit(reshaped_label)
        if fine_grained_level == 'Species':

            self.labels_to_parents = dict(zip(encoded_labels['label'],
                                              list(zip(dataset['Family'],
                                                       dataset['Genus']))))

            def agg_labels(x):
                unique_labels = np.unique(x).reshape(-1, 1)
                one_hot_labels = self.one_hot.transform(unique_labels)
                return list(np.sum(one_hot_labels, axis=0))

            groupby = dataset.groupby(['Family'])['label']
            self.family_to_species = groupby.agg(agg_labels).to_dict()
            groupby = dataset.groupby(['Genus'])['label']
            self.genus_to_species = groupby.agg(agg_labels).to_dict()

        elif fine_grained_level == 'Genus':
            self.labels_to_parents = dict(zip(dataset['label'],
                                              list(zip(dataset['Family']))))
            self.one_hot.transform(groupby)
            groupby = dataset.groupby(['label'])['Species']
            self.genus_to_species = groupby.agg(agg_labels).to_dict()

    def get_hierarchy(self, id):
        res = {}

        if self.fine_grained_level == 'Species':
            family, genus = self.labels_to_parents[id]
            res['same_genus'] = np.asarray(self.genus_to_species[genus])
            res['same_family'] = np.asarray(self.family_to_species[family])

        elif self.fine_grained_level == 'Genus':
            family = self.labels_to_parents[id]
            res['same_family'] = np.asarray(self.family_to_species[family])

        return res

class Geo_Encoder():

    def __init__(self, geo_data_path):
        super().__init__()
        with open(geo_data_path, 'rb') as fhandle:
            self.bin_data = pickle.load(fhandle)
        self.bin_distr = {}
    def softmax(self, X):
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X))
        return expo/expo_sum
    def fit(self, le):
        num_species = len(list(le.classes_))
        for bin_ in self.bin_data['bin2species2count']:
            self.bin_distr[bin_] = np.zeros(num_species)
            for sp in self.bin_data['bin2species2count'][bin_]:
                label = le.transform([sp])
                self.bin_distr[bin_][label] = self.bin_data['bin2species2count'][bin_][sp]
        
        for row in self.bin_distr:
            self.bin_distr[row] = self.bin_distr[row]/self.bin_distr[row].sum()

    def transform(self, df):
        xs = df['Latitude'].to_numpy()
        ys = df['Longitude'].to_numpy()

        cell_row = np.digitize(ys, self.bin_data['y_boundries'])-1
        cell_column = np.digitize(xs, self.bin_data['x_boundries'])-1
        locations = []
        for i in range(len(xs)):
            curr_xs = xs[i]
            curr_ys = ys[i]
            if curr_xs < self.bin_data['X_min'] or curr_xs > self.bin_data['X_max']:
                locations.append(np.nan)
            elif curr_ys < self.bin_data['Y_min'] or curr_ys > self.bin_data['Y_max']:
                locations.append(np.nan)
            else:
                cell_row = np.digitize(curr_ys, self.bin_data['y_boundries'])-1
                cell_column = np.digitize(curr_xs, self.bin_data['x_boundries'])-1
                curr_bin = self.bin_data['bins'][cell_row, cell_column]
                if curr_bin not in self.bin_distr or self.bin_data['coord2count'][curr_bin] < 10:
                    locations.append(np.nan)
                else:
                    locations.append(curr_bin)
        return locations 


class ButterflyDataset(object):
    '''
    The ButterflyDataset instance inherits from the standard
    torch.utils.data.Dataset, it corresponds to the e-butterfly
    dataset structure.

    Parameters
    ----------
    csv_root : str
        path to the info main folder

    info_file : str
        name of the file containing the images info, it must be the
        relative path from root

    image_dir : str
        path to the directory containing the images

    fine_grained_level: str
        the level for the annotation (ie either Species
        or Genus) it should match the column names
        of the info.csv file

    split_name : str
        Either 'train', 'val' or 'test' corresponding to the data split name.

    label_encoder : sklearn.preprocessing.LabelEncoder
        The label encoder for the label of the images (must be
         passed in argument for the test and validation set)

    preprocessing : function
        function to use in data normalization and augmentation

    drop_zero_shot : boolean
        whether to drop the element from the test set that
        have never been seen in the train set

    Attributes
    ----------
    root : str
        path to the data main folder which contains the
        images in ob_with_images and the annotations in a .csv file

    samples : pandas.Dataframe
        the Dataframe containing the information about the
        different images of the dataset

    label_encoder : sklearn.preprocessing.LabelEncoder
        the label encoder for the label of the images

    fine_grained_level : str
        the level for the annotation (ie either Species or Genus)
        it should match the column names
        of the info.csv file

    augmentation : function
        the augmentation applied to the images for data-augmentation
    '''

    def __init__(self, config, csv_root, image_dir, info_file,
                 fine_grained_level='Species',
                 split_name='train',
                 label_encoder=None, preprocessing=None,
                 drop_zero_shot=True,
                 masks_dir=None, geo_encoder=None):
        self.csv_root = csv_root
        self.image_dir = image_dir
        self.split_name = split_name
        self.masks_dir = masks_dir
        try:
            assert self.split_name in ['train', 'val', 'test']
        except AssertionError:
            raise ValueError('The split_name arg should be train, val or test')

       # if not self.split_name == 'val':
        #    try:
         #       assert self.split_name in info_file
          #  except AssertionError:
           #     err = str('split name ({}) and the info '
            #              'file ({}) are not matching'.format(self.split_name,
             #                                                 info_file))
              #  raise ValueError(err)

        try:
            assert os.path.exists(self.image_dir)
        except AssertionError:
            err = str('The images root folder is '
                      'not matching the example organization')
            raise ValueError(err)

        self.samples = pd.read_csv(os.path.join(csv_root, info_file))
        if label_encoder is None:
            # Build the encoder because we are in the training
            self.label_encoder = Encoder()
            self.label_encoder.fit(self.samples, fine_grained_level)
            if config.geo_bins:
                self.geo_encoder = Geo_Encoder(config.geo_bins)
                self.geo_encoder.fit(self.label_encoder.le)
        else:
            if drop_zero_shot:
                # We might have issues with observation only resent in the test set
                # we keep only the samples that are already encoded
                classes = label_encoder.classes_
                idx2keep = self.samples[fine_grained_level]
                idx2keep = idx2keep.isin(classes)
                self.samples = self.samples[idx2keep]
            # Don't fit it as we are in val or test
            self.label_encoder = label_encoder
            if config.geo_bins:
                self.geo_encoder = geo_encoder

        self.samples = self.samples.dropna(subset = ['Latitude', 'Longitude', 'Date Observed'])
        new_label = self.label_encoder.transform(
            self.samples[fine_grained_level])
        encoded_labels = pd.DataFrame(new_label,
                                      columns=['label'],
                                      index=self.samples.index)
        if config.geo_bins:
            new_geo_label = self.geo_encoder.transform(self.samples)
            encoded_geo_labels = pd.DataFrame(new_geo_label, 
                                          columns=['bin'], 
                                          index=self.samples.index)
            self.samples = pd.concat([self.samples, encoded_labels, encoded_geo_labels], axis=1)
            self.samples = self.samples[self.samples['bin'].notna()]
        else:
            self.samples = pd.concat([self.samples, encoded_labels], axis=1)
        self.num_classes = self.label_encoder.classes_.shape[0]
        self.samples = self.samples.loc[self.samples['split']
                                        == self.split_name]
        
        if False and self.split_name == "train" and config.upsample:
            print("UPSAMPLING DATA>>>>>>>>>>>>>>>>")
            train_labels = self.samples['label'].values
            train_feats = self.samples['path'].values
            label2feats = {}
            for i in range(len(train_labels)):
                if train_labels[i].item() not in label2feats:
                    label2feats[train_labels[i].item()] = []
                label2feats[train_labels[i].item()].append(train_feats[i])
            np.random.seed(42)
            train_feats_upsampled = []
            train_classes = []
            for label in sorted(label2feats):
                num_samples = len(label2feats[label])
                if num_samples < 50:
                    curr_train_feats = np.random.choice(label2feats[label], size=50, replace=True)
                else:
                    curr_train_feats = (label2feats[label])
                for row in curr_train_feats:
                    train_feats_upsampled.append([row, label])
            np.random.shuffle(train_feats_upsampled)
            self.samples = pd.DataFrame(train_feats_upsampled, columns=['path', 'label'])
        print("NUM_{}_SAMPLES:::{}".format(self.split_name, len(self.samples.index)))
        self.occ = self.samples['label'].value_counts()
        self.augment = False
        if hasattr(preprocessing, 'normalization'):
            self.normalization = Preprocessing(preprocessing.normalization)
        else:
            self.normalization = None
        augment_is_def = hasattr(preprocessing, 'augmentation')
        if self.split_name == 'train' and augment_is_def:
            self.augment = True
            self.augmentation = Preprocessing(preprocessing.augmentation)
        else:
            self.augmentation = None
    def preprocess_img(self, img):
        if self.masks_dir is not None:
            try:
                infos['mask'] = np.load(os.path.join(self.masks_dir,
                                                     path.split('.')[0] +'.npy'))
            except FileNotFoundError:
                infos['mask'] = np.zeros(img.shape[:2])
            infos['mask'] = infos['mask'].astype('float')
            infos['mask'] = np.expand_dims(infos['mask'], axis=-1)
            infos['mask'] = np.repeat(infos['mask'], 3, axis=-1)
        if self.augment and self.augmentation is not None:
            img = self.augmentation(img)
            if self.masks_dir is not None:
                infos['mask'] = self.augmentation(infos['mask'])
        if self.normalization is not None:
            img = self.normalization(img)
            if self.masks_dir is not None:
                infos['mask'] = self.normalization(infos['mask'])
                infos['mask'] = infos['mask'].astype('int')
        if self.masks_dir is not None:
            infos['mask'] = infos['mask'].astype('int')[:, :, 0]
        img = np.transpose(img, (2, 0, 1)).copy().astype(
            np.float32, copy=True) / 255
        return img
    def __getitem__(self, idx):
        # load images and masks
        infos = self.samples.iloc[idx].to_dict()
        infos.update(self.label_encoder.get_hierarchy(infos['label']))
        path = infos['path']
        img_path = os.path.join(self.image_dir, path)
        img = cv2.imread(img_path)[:, :, ::-1]
        img = self.preprocess_img(img)

        return img, infos

    def __len__(self):
        return self.samples.shape[0]
