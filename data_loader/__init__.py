import torch
import numpy as np

from data_loader.butterfly import ButterflyDataset


def setup_dataset(config, val_only=False, test_only=False, debug=False,
                  return_dataset=False, return_dataloader=True):
    """
    Input :
        - the dataset part of the global parsed config file

    Output :
        - train data_loader
        - validation data_loader
    """
    alpha = config.training.class_weights_alpha

    try:
        weighting_type = config.training.weighting_type
    except AttributeError:
        print('By default the weighting sampler type is cast to linear')
        weighting_type = 'linear'

    len_debug_train = 1000
    len_debug_dev = 100

    csv_root = config.dataset.csv_root
    image_dir = config.dataset.image_dir
    try:
        masks_dir = config.dataset.masks_dir
    except AttributeError:
        masks_dir = None
    fine_grained_level = config.dataset.fine_grained_level
    preprocessing = config.dataset.preprocessing
    train_dataset = ButterflyDataset(config, csv_root, image_dir,
                                     config.dataset.train_info_file,
                                     fine_grained_level,
                                     split_name='train',
                                     preprocessing=preprocessing,
                                     masks_dir=masks_dir)
    if config.geo_bins:
        geo_encoder = train_dataset.geo_encoder
    else:
        geo_encoder = None
    val_dataset = ButterflyDataset(config, csv_root, image_dir,
                                   config.dataset.train_info_file,
                                   fine_grained_level,
                                   split_name='val',
                                   label_encoder=train_dataset.label_encoder,
                                   preprocessing=preprocessing,
                                   masks_dir=masks_dir, geo_encoder=geo_encoder)

    if debug:
        train_dataset.samples = train_dataset.samples.head(len_debug_train)
        val_dataset.samples = val_dataset.samples.head(len_debug_dev)

    if val_only:
        if return_dataloader:
            val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=config.training.batch_size,
                                                      num_workers=config.runtime.num_workers,
                                                      drop_last=False)
        
            return val_data_loader
        else:
            return val_dataset

    if test_only:
        test_dataset = ButterflyDataset(config, csv_root, image_dir,
                                        config.dataset.test_info_file,
                                        fine_grained_level,
                                        split_name='test',
                                        label_encoder=train_dataset.label_encoder,
                                        preprocessing=config.dataset.preprocessing, geo_encoder=geo_encoder)
    
        
        if debug:
            test_dataset.samples = test_dataset.samples.head(len_debug_dev)
        if return_dataloader:
            return torch.utils.data.DataLoader(test_dataset, batch_size=config.training.batch_size,
                                           num_workers=config.runtime.num_workers, drop_last=False)
        else:
            return test_dataset

    if weighting_type == 'linear':
        weights = (
            (1/train_dataset.occ[train_dataset.samples['label']])**(alpha)).tolist()

    elif weighting_type == 'log':
        weights = (
            (1/np.log(1+train_dataset.occ[train_dataset.samples['label']])**(alpha)).tolist())

    elif weighting_type == 'balanced_linear':
        weights = (
            (1/train_dataset.occ[train_dataset.samples['label']])**(alpha)).tolist()
        weights = np.array(weights)
        coef = config.training.balancing_coefficient
        weights[0] = coef
        weights[1:] = (1 - coef) * weights[1:] / np.sum(weights[1:])

    assert config.model.num_classes == train_dataset.num_classes, 'model num_classes {} |  train_dataset num_classes {}'.format(
        config.model.num_classes, train_dataset.num_classes)

    # Create the data_loaders according to the weights
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights),
                                                     replacement=True)
    if return_dataset:
        return train_dataset, val_dataset, sampler
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler,
                                                        batch_size=config.training.batch_size,
                                                        num_workers=config.runtime.num_workers,
                                                        drop_last=True)
        if not val_only:
            val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.training.batch_size,
                                                          num_workers=config.runtime.num_workers,
                                                          drop_last=False)

        if config.geo_bins and return_dataloader:
            return train_data_loader, val_data_loader, train_dataset.geo_encoder.bin_distr
        elif return_dataloader:
            return train_data_loader, val_data_loader
        else:
            return train_dataset, val_dataset, len(list(train_dataset.label_encoder.classes_))
