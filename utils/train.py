from comet_ml import Experiment
import magicpath
from data_loader import setup_dataset
from models import get_model
from utils.utilities import ObjFromDict
from utils.loss import Cross_Entropy_Loss, Guided_Attention_Loss
from utils.loss import Hierarchical_Cross_Entropy_Loss
import argparse
import datetime
import json
import os
import geo_models
from mega_model import MegaModel

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
assert magicpath
import copy
experiment = Experiment(api_key="IwUoq6LzlxwDePJx1egspBLcQ",
                        project_name="butterflai", workspace="martaskrt")


def train_one_epoch(config, model, optimizer, data_loader, loss_object, device,
                    epoch, writer, freq_print=10000,
                    metrics=['top3_acc', 'accuracy']):
    model.train()
    avg_loss = 0
    correct = np.zeros(model.num_classes)
    correct_top3 = np.zeros(model.num_classes)
    count_label = np.zeros(model.num_classes)  # only for micro accuracy
    top1_epoch_acc = top3_epoch_acc = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        loss, output = loss_object.compute_loss(config, model, data, target, train=True)
        target['label'] = target['label'].to(device)
        loss.backward()
        optimizer.step()
        
        if avg_loss == 0:
            avg_loss = loss.item()
        avg_loss = 0.9 * avg_loss + 0.1 * loss.item()
        
        if freq_print is not None and batch_idx % freq_print == 0:
            tot_idx = batch_idx * len(data)
            n_tot = len(data_loader.dataset)
            progress = 100. * batch_idx / len(data_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'avg Loss: {:.6f}\t Loss: {:.6f}'.format(epoch, tot_idx,
                                                           n_tot, progress,
                                                           avg_loss,
                                                           loss.item()))
        
       
        label = target['label'].cpu()
        #count_label[label] += 1
        prob_pred_top3, class_pred_top3 = torch.topk(output, 3, dim=1,
                                                     largest=True, sorted=True,
                                                     out=None)
        is_top_3 = class_pred_top3.eq(target['label'].view(-1, 1)).sum(dim=-1)
        is_top_3 = np.squeeze(is_top_3.cpu().numpy())
        #correct_top3[label] += is_top_3
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        for i in range(len(label)):
            curr_label = label[i].item()
            count_label[curr_label] += 1
            if pred[i] == curr_label:
                correct[curr_label] += 1
            if curr_label in class_pred_top3[i]:
                correct_top3[curr_label] += 1
        
        is_correct = pred.eq(target['label'].view_as(pred))
        is_correct = np.squeeze(is_correct.cpu().numpy())
        #correct[label] += is_correct
        # the accuracy are first averaged by batch
    top1_epoch_acc = np.sum(correct) / np.sum(count_label)
    top3_epoch_acc = np.sum(correct_top3) / np.sum(count_label)
    correct = correct[count_label > 0]
    correct_top3 = correct_top3[count_label > 0]
    count_label = count_label[count_label > 0]
    micro_top1_epoch_acc = np.mean(correct / count_label)
    micro_top3_epoch_acc = np.mean(correct_top3/count_label)
    loss_object.on_epoch_end()
    writer.add_scalar('train_epoch_loss', avg_loss, epoch)
    writer.add_scalar('train_epoch_acc_top1', top1_epoch_acc, epoch)
    writer.add_scalar('train_epoch_acc_top3', top3_epoch_acc, epoch)
    writer.add_scalar('train_epoch_micro_acc_top1',
                      micro_top1_epoch_acc, epoch)
    writer.add_scalar('train_epoch_micro_acc_top3',
                      micro_top3_epoch_acc, epoch)
    print('Train Epoch: {} \t avg Loss: {:.4e}\t train top 1 acc: {:.4%}'
          ' (macro) / {:.4%} (micro) \t '
          'train epoch top 3 acc: {:.4%} '
          '(macro) / {:.4%} (micro)'.format(epoch,
                                            avg_loss,
                                            top1_epoch_acc,
                                            micro_top1_epoch_acc,
                                            top3_epoch_acc,
                                            micro_top3_epoch_acc))

    experiment.log_metric("train_loss", avg_loss, step=epoch)
    experiment.log_metric("train_macro_top1", top1_epoch_acc, step=epoch)
    experiment.log_metric("train_micro_top1", micro_top1_epoch_acc, step=epoch)
    experiment.log_metric("train_macro_top3", top3_epoch_acc, step=epoch)
    experiment.log_metric("train_micro_top3", micro_top3_epoch_acc, step=epoch)
    return writer, loss_object


def evaluate(config, model, data_loader, loss_object,
             device, epoch, writer, metrics=['top3_acc', 'accuracy']):
    model.eval()

    with torch.no_grad():
        validation_loss = 0
        correct = np.zeros(model.num_classes)
        correct_top3 = np.zeros(model.num_classes)
        count_label = np.zeros(model.num_classes)  # only for micro accuracy
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            # Compute the scores
            loss, output = loss_object.compute_loss(config, model, data, target, a=a, b=b, c=c)
            target['label'] = target['label'].to(device)
            validation_loss += loss.item()

            label = target['label'].cpu()
            #count_label[label] += 1
            prob_pred_top3, class_pred_top3 = torch.topk(output, 3, dim=1,
                                                         largest=True,
                                                         sorted=True,
                                                         out=None)
            is_top_3 = class_pred_top3.eq(
                target['label'].view(-1, 1)).sum(dim=-1)
            is_top_3 = np.squeeze(is_top_3.cpu().numpy())
            #correct_top3[label] += is_top_3
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(len(label)):
                curr_label = label[i].item()
                count_label[curr_label] += 1
                if pred[i] == curr_label:
                    correct[curr_label] += 1
                if curr_label in class_pred_top3[i]:
                    correct_top3[curr_label] += 1
                    
            is_correct = pred.eq(target['label'].view_as(pred))
            is_correct = np.squeeze(is_correct.cpu().numpy())
            #correct[label] += is_correct
        val_loss = validation_loss/len(data_loader)
        top1_acc = np.sum(correct) / np.sum(count_label)
        top3_acc = np.sum(correct_top3) / np.sum(count_label)
        correct = correct[count_label > 0]
        correct_top3 = correct_top3[count_label > 0]
        count_label = count_label[count_label > 0]
        micro_top1_acc = np.mean(correct / count_label)
        micro_top3_acc = np.mean(correct_top3/count_label)
        writer.add_scalar('val_epoch_loss', val_loss, epoch)
        writer.add_scalar('val_epoch_macro_acc_top1', top1_acc, epoch)
        writer.add_scalar('val_epoch_macro_acc_top3', top3_acc, epoch)
        writer.add_scalar('val_epoch_micro_acc_top1', micro_top1_acc, epoch)
        writer.add_scalar('val_epoch_micro_acc_top3', micro_top3_acc, epoch)
        eval_score = {}
        eval_score['top3_acc'] = top3_acc
        eval_score['accuracy'] = top1_acc
        eval_score['micro_accuracy'] = micro_top1_acc
        eval_score['micro_top3_acc'] = micro_top3_acc
        print('epoch : {} val_loss : {:.4e} | top1_acc {:.4%} (macro) /  '
              '{:.4%} (micro) | top3_acc {:.4%} (macro) '
              '/ {:.4%} (micro)'.format(epoch, val_loss,
                                        top1_acc, micro_top1_acc,
                                        top3_acc, micro_top3_acc))
        loss_object.on_epoch_end()
        experiment.log_metric("val_loss", val_loss, step=epoch)
        experiment.log_metric("val_macro_top1", top1_acc, step=epoch)
        experiment.log_metric("val_micro_top1", micro_top1_acc, step=epoch)
        experiment.log_metric("val_macro_top3", top3_acc, step=epoch)
        experiment.log_metric("val_micro_top3", micro_top3_acc, step=epoch)
    return writer, eval_score, loss_object


def update_sampler(config, model, dataset, loss_object, device):
    batch_size = config.training.batch_size
    num_workers = config.runtime.num_workers
    dataset.aumgnent = False
    data_loader = DataLoader(dataset, shuffle=False,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             drop_last=False)

    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # Compute the scores
            loss, output = loss_object.compute_loss(
                model, data, target, reduce=False)
            target['label'] = target['label'].to(device)
            # Loss function needs to keep the batch dimension
            batch_loss = loss.to('cpu').numpy()
            result = pd.DataFrame(
                {'label': target['label'].cpu().numpy(),
                 'loss': np.squeeze(batch_loss)})
            results.append(result)
    loss_object.on_epoch_end()
    results = pd.concat(results)
    results = results.groupby(['label']).agg({"loss": 'mean'})
    sample_class_error = results.loc[dataset.samples['label']]
    sample_class_error = np.asarray((sample_class_error).values)
    sample_class_error = (sample_class_error - sample_class_error.min())
    sample_class_error = sample_class_error / \
        (sample_class_error.max() - sample_class_error.min())
    weights = sample_class_error
    sampler = torch.utils.data.WeightedRandomSampler(np.squeeze(weights),
                                                     num_samples=len(weights),
                                                     replacement=True)
    dataset.aumgnent = True
    return sampler, loss_object


def main(raw_args=None):
    """
    Main function to run the code. Can either take the raw_args
    in argument or get the arguments from the config_file.
    """

    # -----------------------------------------------------------------------------------------
    # First, set the parameters of the function, including the
    #  config file, log directory and the seed.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None,
                        type=str, help='path to the config'
                        'file for the training')
    parser.add_argument('--logdir', required=True,
                        type=str, help='path to the directory'
                        'containing all run folders')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='dataloader threads')
    parser.add_argument('--seed', type=int,
                        default=np.random.randint(2**32 - 1),
                        help='the seed for reproducing experiments')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='the path to the log dir of the '
                        'training to resume')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='whether to debug or not')
    parser.add_argument('--freq_print', type=int, default=None,
                        help='frequency for the printing of the'
                        ' loss during training')
    parser.add_argument('--clean_data', action='store_true')
    parser.add_argument("--upsample", action='store_true')
    args = parser.parse_args(raw_args)
    err = 'You should provide a config_file or a resume_dir'
    assert args.config_file is not None or args.resume_dir is not None, err
    

    if args.resume_dir is None:
        resumed = False
        # Allows to quickly know if you set the wrong seed
        print("SEED used is ", args.seed)
        torch.manual_seed(args.seed)  # the seed for every torch calculus
        np.random.seed(args.seed)  # the seed for every numpy calculus
        # -----------------------------------------------------------------------------------------
        # Prepare the log by adding the config with runtime and seed
        with open(args.config_file) as json_file:
            config_dict = json.load(json_file)

        try:
            print('loss type is', config_dict['training']['loss']['type'])
        except KeyError:
            loss_missing = 'loss' not in config_dict['training'].keys()
            if loss_missing:
                print('loss type is defaulted to classification_only')
                config_dict['training']['loss'] = {
                    'type': 'classification_only'}
            type_missing = 'type' not in config_dict['training']['loss'].keys()
            if type_missing:
                print('loss type is defaulted to classification_only')
                config_dict['training']['loss'] = {
                    'type': 'classification_only'}

        if 'finetune_from' not in config_dict['model'].keys():
            print('finetune is defaulted to false')
            config_dict['model']['finetune_from'] = False

        config_dict['runtime'] = {}
        config_dict['runtime']['num_workers'] = args.num_workers
        config_dict['dataset']['num_workers'] = args.num_workers
        config_dict['runtime']['SEED'] = args.seed

        if not os.path.exists(args.logdir):
            os.mkdir(args.logdir)

        time = datetime.datetime.today()
        log_id = '{}_{}h{}min'.format(time.date(), time.hour, time.minute)
        log_path = os.path.join(args.logdir, log_id)
        i = 0
        created = False
        while not created:
            try:
                os.mkdir(log_path)
                created = True
            except OSError:
                i += 1
                log_id = '{}_{}h{}min_{}'.format(
                    time.date(), time.hour, time.minute, i)
                log_path = os.path.join(args.logdir, log_id)
        with open(os.path.join(log_path, 'config.json'), 'w') as file:
            json.dump(config_dict, file, indent=4)

    else:  # else we resume the training from the resume dir
        resumed = True
        args.config_file = os.path.join(args.resume_dir, "config.json")
        log_path = args.resume_dir
        # load the used config_file
        with open(args.config_file) as json_file:
            config_dict = json.load(json_file)
        # reset the old seed
        torch.manual_seed(config_dict['runtime']['SEED'])
        np.random.seed(config_dict['runtime']['SEED'])

    # -----------------------------------------------------------------------------------------
    # Get the parameters according to the configuration
    config_dict['encode_geo_bins'] = False
    config_dict['geo'] = None
    config_dict['geo_bins'] = False
    config_dict['clean_data'] = args.clean_data
    config_dict['upsample'] = args.upsample
    config = ObjFromDict(config_dict)
    experiment.set_name(args.logdir)

    model = get_model(config.model, config.training.loss.type)
    train_dataloader, val_dataloader = setup_dataset(config, debug=args.debug)
    class_occurences = train_dataloader.dataset.occ
    # Compute on gpu or cpu
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Set the variables
    if config.training.loss.type == 'classification_only':
        loss_object = Cross_Entropy_Loss()

    elif config.training.loss.type == "hierarchical":
        loss_dict = config_dict['training']['loss']['params']
        batch_size = config.training.batch_size
        loss_object = Hierarchical_Cross_Entropy_Loss(batch_size,
                                                      len(class_occurences),
                                                      **loss_dict)
    elif config.training.loss.type == "guided_attention":
        loss_dict = config_dict['training']['loss']['params']
        batch_size = config.training.batch_size
        loss_dict["device"] = device
        loss_object = Guided_Attention_Loss(**loss_dict)

    else:
        raise NotImplementedError('loss type should be '
                                  'classification_only or geometric')
    # trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer and learning rate
    real_lr = 10**(-config.optimizer.learning_rate)
    real_decay = 10**(-config.optimizer.weight_decay)
    optimizer = torch.optim.Adam(params, lr=real_lr,
                                 weight_decay=real_decay)
    step_size = config.optimizer.lr_scheduler.step_size
    gamma = config.optimizer.lr_scheduler.gamma
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                  gamma=gamma)
    # tensorboard logs
    writer = SummaryWriter(log_path)
    best_scores = {}
    metrics = ['top3_acc', 'accuracy']

    for metric in metrics+['micro_accuracy', 'micro_top3_acc']:
        best_scores[metric] = -1
    start_epoch = 0
    config.use_imgs = False
    if resumed:
        training_state = torch.load(os.path.join(log_path, "training_state.pth.tar"), map_location=torch.device('cpu'))
        model.load_state_dict(training_state['state_dict'])

        start_epoch = training_state['epoch']
        optimizer.load_state_dict(training_state['optimizer'])
        lr_scheduler.load_state_dict(training_state['lr_scheduler'])
        best_scores = training_state['best_scores']
        model = model.to(device)


    start_epoch = 0
    print(device)
    for epoch in tqdm(range(start_epoch, config.training.epochs)):

        writer, loss_object = train_one_epoch(config, model, optimizer,
                                              train_dataloader, loss_object,
                                              device, epoch, writer,
                                              freq_print=args.freq_print,
                                              metrics=metrics)
        eval_args = [config, model, val_dataloader, loss_object, device,
                     epoch, writer]
        writer, eval_score, loss_object = evaluate(*eval_args, metrics=metrics)
        lr_scheduler.step()
        # save the best model for each metric
        for metric in list(eval_score.keys()):
            if eval_score[metric] > best_scores[metric]:
                torch.save(model.state_dict(), os.path.join(
                    log_path, 'best_{}.pth'.format(metric)))
                best_scores[metric] = eval_score[metric]
        # save the state of the training
        training_state = {'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'best_scores': best_scores}
        torch.save(training_state, os.path.join(
            log_path, "training_state.pth.tar"))
    writer.close()

    return best_scores


if __name__ == '__main__':
    main()
