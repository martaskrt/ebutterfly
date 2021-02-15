import os
import time
import copy
import argparse
import magicpath
import json

import tqdm
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms

from models import load_model
from data_loader import setup_dataset
from utilities import ObjFromDict

assert magicpath
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def infer(model, dataloader, output_path, device='cuda'):
    '''Recover the results of the model on the test dataset'''
    assert output_path.endswith('.csv')
    model = model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader)):
            data = data.to(device)
            output = model(data)
            prob_pred_top3, class_pred_top3 = torch.topk(
                output, 3, dim=1, largest=True, sorted=True, out=None)
            top3array = pd.DataFrame(np.squeeze(
                class_pred_top3.to('cpu').numpy()))
            target['predicted_1'] = top3array[0]
            target['predicted_2'] = top3array[1]
            target['predicted_3'] = top3array[2]
            try:
                del target['same_family']
                del target['same_genus']
            except KeyError:
                pass
            results.append(pd.DataFrame(target, index=None))
    result = pd.concat(results)
    result.to_csv(output_path, index=None)
    return result


def get_embedding(model, dataloader, output_path,
                  output_path_label, device='cuda'):
    '''Build the last embedding from the model deprived from the classifier'''
    assert output_path.endswith('.npy')
    model = model.to(device)
    model.eval()
    embedding = torch.nn.Sequential(*list(model.children())[:-1])
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader)):
            data = data.to(device)
            output = embedding(data).cpu().numpy()
            embeddings.append(output)
            labels.append(copy.deepcopy(target['label'].cpu().numpy()))
    labels = np.concatenate(labels)
    embeddings = np.concatenate(embeddings)
    embeddings = np.squeeze(embeddings)
    np.save(output_path, embeddings)
    np.save(output_path_label, labels)
    return embeddings, labels

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def get_accuracy(results, k, micro=False):
    """
    Function that computes the top k accuracy from a dataframe
    with a column label and the top k predicted
    """
    results['correct'] = 0
    for i in range(1, k + 1):
        pred_col = 'predicted_{}'.format(i)
        results['correct'] += (results['label'] == results[pred_col])
    if micro:
        acc = results.groupby(['label']).agg({"correct": 'mean'})
        acc = acc['correct'].mean()
    else:
        acc = results['correct'].mean()
    return acc


def create_acc_per_occ_curve(run_dir, unique_occ, res, k):
    acc_per_occ = []
    for n in unique_occ:
        acc_per_occ.append(get_accuracy(res.loc[(res['occ'] == n)], k))
    fg, ax = plt.subplots(1, 1)
    ax.set_title('top {} accuracy per number of occurences '
                 'in the training set'.format(k))
    ax.set_xlabel('number of example in the training set')
    ax.set_ylabel('top {} accuracy'.format(k))
    ax.scatter(unique_occ, acc_per_occ)
    fg.savefig(os.path.join(run_dir, 'top{}_accuracy_per_occ.jpg'.format(k)))


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def get_tsne(embeddings, output_path_tsne,
             output_path_pca_plot, n_comp_pca=10):
    '''Build a t-SNE visualization of the embedding
    after computing a first PCA'''
    assert output_path_tsne.endswith('.npy')
    print('starting tsne embedding')
    start = time.time()
    pca = PCA(n_comp_pca)
    small_embeddings = pca.fit_transform(embeddings)
    fg, ax = plt.subplots(1, 1)
    ax.set_title('PCA explained variance before tsne')
    ax.plot(pca.explained_variance_ratio_)
    fg.savefig(output_path_pca_plot)
    tsne_embedded = TSNE(n_components=2,).fit_transform(small_embeddings)
    np.save(output_path_tsne, tsne_embedded)
    print('time taken for the tsne embedding {:.1f}'.format(
        time.time() - start))
    return tsne_embedded

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def compute_grad_weights(model, last_conv_layer, target_class, sample, device):
    '''Computes the grad part to specialize the computation of the CAM'''
    store = {}

    def register_convmap(self, in_map, out_map):
        store["conv_maps"] = out_map

    def register_grads(self, in_grads, out_grads):
        store["grads"] = out_grads

    handle_fw = last_conv_layer.register_forward_hook(register_convmap)
    handle_bw = last_conv_layer.register_backward_hook(register_grads)

    try:
        output = model(sample)

        one_hot = torch.FloatTensor(1, model.num_classes)
        one_hot.zero_()
        one_hot[:, target_class] = 1
        output = output.to(device)
        one_hot = one_hot.to(device)
        one_hot = torch.sum(one_hot * output)

        model.zero_grad()
        one_hot.backward()

    finally:
        handle_fw.remove()
        handle_bw.remove()

    conv_grads = store["grads"]
    grad_weights = torch.mean(conv_grads[0], dim=[2, 3])

    return grad_weights, store["conv_maps"]


def compute_heatmap(conv_maps, grad_weights, model_out_size, device,
                    interpolation_mode='bilinear',
                    original_out_size=(400, 400)):
    '''Build the grad-CAM taking the grad weights in argument'''
    with torch.no_grad():
        for i in range(conv_maps.shape[1]):
            conv_maps[:, i] *= grad_weights[:, i]
            conv_maps = conv_maps.to(device)
            coarse_heatmap = torch.relu(conv_maps.sum(dim=1)).detach()
        coarse_heatmap -= coarse_heatmap.min()
        coarse_heatmap /= coarse_heatmap.max()

        # resnets have an output in 13x13 with input size of 400x400
        heatmap = coarse_heatmap.view(1, -1, model_out_size, model_out_size)
        heatmap = torch.nn.functional.interpolate(heatmap,
                                                  size=original_out_size,
                                                  mode=interpolation_mode)
    return coarse_heatmap, heatmap

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def main(raw_args=None):
    '''
    Generate a HTML file that contains a thorough analysis of the performance
    of the algorithm. It computes :
     - Accuracy per occurence in the training set
     - t-SNE visualization of the embedding
     - Grad-Class Activation Map on a few examples
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        type=str, help='path to the run directory '
                        'to perform the error analysis on')
    parser.add_argument('--results_csv_file', required=True,
                        type=str, help='path to the csv file '
                        'containing all the experiments results')
    parser.add_argument('--recompute', action='store_true', default=False,
                        help='whether to recompute the '
                        'embedding if already in the folder')
    parser.add_argument('--split', type=str, default='test',
                        help='specify if you want to run the '
                        'error analysis on test, val or train')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='define a new batch size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='define a new number of workers')
    parser.add_argument('--acc_per_occ', action='store_true', default=False,
                        help='Whether to plot the accuracy per occurence '
                        'curve or not')
    parser.add_argument('--t_sne', action='store_true', default=False,
                        help='Whether to plot t-SNE visualizations '
                        'of the embeddings')
    parser.add_argument('--grad_cam', action='store_true', default=False,
                        help='Whether to plot grad-CAM visualization '
                        'on some examples')
    args = parser.parse_args(raw_args)

    recompute = args.recompute
    run_dir = args.run_dir
    model, config = load_model(run_dir, metric='top3_acc')

    highlighted_classes = []
    while len(highlighted_classes) < 3:
        selected_class = np.random.randint(1, config.model.num_classes)
        if selected_class not in highlighted_classes:
            highlighted_classes.append(selected_class)

    output_dir = os.path.join(run_dir, 'error_analysis_{}'.format(args.split))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if args.batch_size is None:
        config.training.batch_size = 32
    else:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.runtime.num_workers = args.num_workers

    train_dataloader, val_dataloader = setup_dataset(config)
    if args.split == 'test':
        dataloader = setup_dataset(config, test_only=True)

    elif args.split == 'val':
        dataloader = val_dataloader

    elif args.split == 'train':
        dataloader = train_dataloader

    if not recompute:
        try:
            result = pd.read_csv(os.path.join(
                output_dir, '{}_pred.csv'.format(args.split)))
            print('Predictions recovered from previous inference')
        except FileNotFoundError:
            print(
                'Not able to recover predictions from previous inference'
                ', starting inference again ...')
            result = infer(model, dataloader, os.path.join(
                output_dir, '{}_pred.csv'.format(args.split)))
    else:
        result = infer(model, dataloader, os.path.join(
            output_dir, '{}_pred.csv'.format(args.split)))

    occ = train_dataloader.dataset.occ
    occ = pd.DataFrame(occ, index=None).reset_index()
    occ = occ.rename({'index': 'label', 'label': 'occ'}, axis=1)
    results_with_occ = pd.merge(result, occ, on='label', how='inner')
    unique_occ = occ['occ'].unique()

    if args.acc_per_occ:
        create_acc_per_occ_curve(output_dir, unique_occ, results_with_occ, 1)
        create_acc_per_occ_curve(output_dir, unique_occ, results_with_occ, 3)

    label_encoder = train_dataloader.dataset.label_encoder
    labels_to_parents = label_encoder.labels_to_parents
    fine_grained_level = label_encoder.fine_grained_level
    final_pred = result['predicted_1']

    hierarchical_acc = {'Hierarchical recovered Family macro score': None,
                        'Hierarchical recovered Family micro score': None,
                        'Hierarchical recovered Genus macro score': None,
                        'Hierarchical recovered Genus micro score': None}
    if fine_grained_level == 'Species':
        pred_family = final_pred.copy().apply(
            lambda label: labels_to_parents[label][0])
        pred_genus = final_pred.copy().apply(
            lambda label: labels_to_parents[label][1])
        final_pred = pd.concat([pred_family, pred_genus],
                               axis=1, ignore_index=True)
        final_pred.columns = ['Pred_family', 'Pred_genus']
        res_parents = pd.concat([result, final_pred], axis=1)

        # Now that we have columns with the predicted Genus and Family
        # corresponding to each predicted species.
        # We can recover the corresponding Top1 accuracy :
        res_parents['correct'] = (res_parents['Pred_family']
                                  == res_parents['Family'])

        acc = res_parents.groupby(['Family']).agg({"correct": 'mean'})
        hierarchical_acc['Hierarchical recovered Family micro score'] = \
            acc['correct'].mean()

        hierarchical_acc['Hierarchical recovered Family macro score'] = \
            res_parents['correct'].mean()

        res_parents['correct'] = (res_parents['Pred_genus']
                                  == res_parents['Genus'])

        acc = res_parents.groupby(['Genus']).agg({"correct": 'mean'})
        hierarchical_acc['Hierarchical recovered Genus micro score'] =\
            acc['correct'].mean()

        hierarchical_acc['Hierarchical recovered Genus macro score'] =\
            res_parents['correct'].mean()

    elif fine_grained_level == 'Genus':
        final_pred = final_pred.copy().apply(
            lambda label: labels_to_parents[label][0])
        final_pred.columns = ['Pred_family']
        res_parents = pd.concat([result, final_pred], axis=1)

        # Now that we have columns with the predicted Genus and Family
        # corresponding to each predicted species.
        # We can recover the corresponding Top1 accuracy :
        res_parents['correct'] = (res_parents['Pred_family']
                                  == res_parents['Family'])

        acc = res_parents.groupby(['Family']).agg({"correct": 'mean'})
        hierarchical_acc['Hierarchical recovered Family micro score'] =\
            acc['correct'].mean()

        hierarchical_acc['Hierarchical recovered Family macro score'] =\
            res_parents['correct'].mean()

    tsne_path = os.path.join(output_dir,
                             'tsne_embedding.npy')
    pca_path = os.path.join(output_dir,
                            'pca_explained_ratio.jpg')
    embedding_path = os.path.join(output_dir, 'embedding.npy')

    labels_path = os.path.join(output_dir, 'labels.npy')
    if args.t_sne:
        if not recompute:
            try:
                embeddings = np.load(embedding_path)
                labels = np.load(labels_path)
                print('Embeddings recovered from previous inference')
                try:
                    tsne_embedding = np.load(tsne_path)
                    print('TSNE embeddings recovered from previous inference')
                except FileNotFoundError:
                    print('Not able to recover TSNE embeddings '
                          'from previous inference')

                    tsne_embedding = get_tsne(embeddings,
                                              tsne_path,
                                              pca_path)
            except FileNotFoundError:
                print('Not able to recover embeddings from previous inference,'
                      ' starting inference again ...')
                embeddings, labels = get_embedding(model, dataloader,
                                                   embedding_path, labels_path)
                tsne_embedding = get_tsne(embeddings, tsne_path, pca_path)
        else:
            embeddings, labels = get_embedding(model,
                                               dataloader,
                                               embedding_path,
                                               labels_path)
            tsne_embedding = get_tsne(embeddings,
                                      tsne_path, pca_path)

        for label in highlighted_classes:
            fg, ax = plt.subplots(1, 1)
            result_class = results_with_occ.loc[
                results_with_occ['label'] == label]
            class_acc = get_accuracy(result_class, 1)
            n_train_samples = occ.loc[occ['label'] == label]['occ'].values[0]

            ax.set_title('TSNE embedding class {} '
                         'vs others \n n train samples : {} | '
                         'top 1 acc : {:.3f}'.format(label,
                                                     n_train_samples,
                                                     class_acc))
            ax.scatter(tsne_embedding[:, 0],
                       tsne_embedding[:, 1],
                       s=0.05,
                       c=1-(labels == label))
            fg.savefig(os.path.join(output_dir,
                                    'tsne_class_{}.jpg'.format(label)))

    if args.grad_cam:
        # Add a Grad-CAM visualization to interpret our model
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        trunc_model = nn.Sequential(*list(model.children())[:-2])
        trunc_model.eval().to(device)
        with torch.no_grad():
            empty = torch.zeros((1, 3, 400, 400))
            empty = empty.to(device)
            model_out_size = trunc_model(empty).shape[-1]

        last_conv_layer = list(model.children())[-3][-1][-1].conv2
        model.to(device)
        for index in range(5):
            sample, infos = dataloader.dataset[index]
            sample = torch.from_numpy(sample).view(1, 3, 400, 400)
            # 'sample' is the preprocessed image

            # Load original image preprocessed without normalization
            try:
                path = infos['file_name']
                img_path = os.path.join(os.path.join(
                    config.dataset.root, 'train_images'), path)
            except KeyError:
                path = infos['path']
                img_path = os.path.join(config.dataset.image_dir, path)
            input_image = Image.open(img_path).convert('RGB')
            input_image = transforms.Resize(400)(input_image)

            with torch.no_grad():
                sample = sample.to(device)
                output = model(sample)
                values, indices = torch.topk(output, 3)

            fig = plt.figure(figsize=(12, 3))
            fig.suptitle("Heatmaps for image #{}".format(index), y=1.10)
            ax = fig.add_subplot(141)
            ax.set_title("Original Image | label : {}".format(infos["label"]))
            ax.imshow(input_image)

            for i, target_class in enumerate(indices[0]):
                weights, conv_maps = compute_grad_weights(
                    model, last_conv_layer, target_class, sample, device)
                coarse_heatmap, heatmap = compute_heatmap(
                    conv_maps, weights, model_out_size, device)
                ax = fig.add_subplot(142 + i)
                title = "Class #{} predicted".format(target_class.item())
                ax.set_title(title)
                ax.imshow(input_image)
                ax.imshow(heatmap.cpu().numpy()[0, 0], cmap="jet", alpha=0.5)

            fig.savefig(os.path.join(output_dir,
                                     'grad_cam_{}.jpg'.format(index)))

    if args.acc_per_occ or args.t_sne or args.grad_cam:
        # Then we create a html file to plot the images as well as some text.
        html_path = os.path.join(output_dir, 'error_analysis.html')
        with open(html_path, 'w') as file:
            file.write('<!DOCTYPE html>\n'
                       '<html lang="fr">\n'
                       '  <head>\n'
                       '    <meta charset="utf-8">\n'
                       '    <title>Error_Analysis | {0}</title>\n'
                       '  </head>\n'
                       '  <body>\n'
                       '    <div> \n'
                       '       <h1> Error_Analysis | {0} </h1>\n'
                       '       <h2> Global Metrics : </h2>\n'
                       '        <p> Acc: {1} <br /> Top3 acc : {2}  <br /> '
                       '       </p>\n'
                       '        <p> micro Acc: {3} <br />'
                       ' micro Top3 acc : {4}<br />'
                       '       </p>\n'
                       '        <p> hierarchical Accuracies : {5} <br />'
                       '       </p>\n'
                       '    </div>\n'.format(run_dir.split('/')[-1],
                                             get_accuracy(result, 1),
                                             get_accuracy(result, 3),
                                             get_accuracy(
                                                 result, 1, micro=True),
                                             get_accuracy(
                                                 result, 3, micro=True),
                                             hierarchical_acc))
            if args.acc_per_occ:
                file.write('    <div> \n'
                           '       <h2> Impact of the number '
                           'of occurences : </h2>\n'
                           '     <img src="top1_accuracy_per_occ.jpg">\n'
                           '     <img src="top3_accuracy_per_occ.jpg">\n'
                           '    </div>\n')
            if args.t_sne:
                file.write('    <div> \n'
                           '       <h2> TSNE Embedding : </h2>\n'
                           '     <img src="pca_explained_ratio.jpg">\n')
                for i in highlighted_classes:
                    file.write(
                        '     <img src="tsne_class_{}.jpg">\n'.format(i))
                file.write('    </div>\n')

            if args.grad_cam:
                file.write('    <div> \n'
                           '       <h2> Grad-CAM visualizations : </h2>\n')
                for index in range(5):
                    file.write('     <img src="grad_cam_{}.jpg">\n'.format(
                        index))
                file.write('    </div>\n')

            file.write('  </body>\n'
                       '</html>\n')

    # Then we only build a csv file containing the results presented above
    # since a html would be pointless for a table of results.
    results_csv_file = args.results_csv_file
    if not os.path.exists(results_csv_file):
        columns = ['Architecture depth', 'Pretraining', 'Loss',
                   'weighting type', 'learning rate', 'gamma', 'step',
                   'weight decay', 'Batch-size', 'Averaging',
                   'Micro_acc_top1', 'Micro_acc_top3',
                   'Macro_acc_top1', 'Macro_acc_top3']\
            + list(hierarchical_acc.keys())\
            + ['Observation', 'Log directory']
        exp_results = pd.DataFrame(columns=columns)
    else:
        exp_results = pd.read_csv(args.results_csv_file)

    with open(os.path.join(run_dir, 'config.json')) as json_file:
        config_dict = json.load(json_file)
    config = ObjFromDict(config_dict)
    if hasattr(config.model, 'depth'):
        architecture_depth = config.model.depth
    else:
        architecture_depth = 50

    if hasattr(config.model, 'finetune_from'):
        pretraining = str(config.model.finetune_from)
    else:
        pretraining = str(config.model.pretrained)

    if hasattr(config.training, 'loss'):
        loss = str(config_dict['training']['loss'])
    else:
        loss = 'classification_only'

    if hasattr(config.training, 'weighting_type'):
        weighting_type = str(config.training.weighting_type)

    else:
        weighting_type = 'linear'
    weighting_type += str(config.training.class_weights_alpha)

    learning_rate = 10**(-config.optimizer.learning_rate)
    gamma = config.optimizer.lr_scheduler.gamma
    step_size = config.optimizer.lr_scheduler.step_size
    weight_decay = 10**(-config.optimizer.weight_decay)
    batch_size = config.training.batch_size

    if hasattr(config.training, 'loss') and\
            hasattr(config.training.loss, 'params') and\
            hasattr(config.training.loss.params, 'averaging_method'):
        averaging = config.training.loss.params.averaging_method
    else:
        averaging = None

    observation = None
    log_directory = run_dir

    values = [architecture_depth, pretraining, loss, weighting_type,
              learning_rate, gamma, step_size, weight_decay, batch_size,
              averaging, get_accuracy(result, 1, micro=True),
              get_accuracy(result, 3, micro=True),
              get_accuracy(result, 1), get_accuracy(result, 3)] \
        + list(hierarchical_acc.values())\
        + [observation, log_directory]

    new_idx_line = len(exp_results)
    exp_results.loc[new_idx_line] = values
    exp_results.to_csv(results_csv_file, index=False)


if __name__ == '__main__':
    main()
