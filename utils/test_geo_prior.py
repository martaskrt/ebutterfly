from comet_ml import Experiment
import magicpath
from data_loader import setup_dataset
from models import get_model
from models.resnet import Resnet
from utils.utilities import ObjFromDict
from utils.loss_geo_prior import Cross_Entropy_Loss, Guided_Attention_Loss
from utils.loss_geo_prior import Hierarchical_Cross_Entropy_Loss
import argparse
import datetime
import json
import os


import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import geo_utils
import train_geo_net as geo_data_prep
import geo_models

assert magicpath
import pickle
    
experiment = Experiment(api_key="INSERT_APIKEY",
                        project_name="butterflai", workspace="INSERT_USERNAME")


def evaluate(config, model, data_loader, loss_object,
             device, epoch, split, writer, bin_distr=None, metrics=['top3_acc', 'accuracy'],
             save_images=False):
    assert split in ["val", "test"]
    model.eval()
    config.num_classes = 601
    config.pretrained = True

    satellite_model = Resnet(config.model, None).to(device)
    training_state = torch.load("20201005_satelliteimg_iNat_upsample/2020-10-05_18h58min/best_accuracy.pth")
    satellite_model.load_state_dict(training_state)
    satellite_model.eval()
    print("loaded satellite model") 
    geo_model = None
    if config.geo_model:
        num_inputs = 4
        if config.elev:
            num_inputs += 1
        if config.dates:
            num_inputs += 2

        if "ebd" in config.geo_model:
            geo_model = geo_models.FCNet(num_inputs=num_inputs, num_classes=model.num_classes, 
                                    num_filts=256, num_users=None, ebd=True, imgs=config.use_imgs).to(device)
        else:
            geo_model = geo_models.FCNet(num_inputs=num_inputs, num_classes=model.num_classes, 
                                    num_filts=256, num_users=None, ebd=False, imgs=config.use_imgs).to(device)
        training_state = torch.load(config.geo_model, map_location=torch.device('cpu'))
        geo_model.load_state_dict(training_state['state_dict'])
        geo_model.eval()

    results = {} # key  = species, value = [count, top1_correct, top3_correct]
    
    ## to get species similarity:
    if os.path.exists("predictions_per_obs.pickle"):
        with open("predictions_per_obs.pickle", 'rb') as fhandle:
            predictions_per_obs = pickle.load(fhandle)
    else:
        predictions_per_obs = {}

    with torch.no_grad():
        validation_loss = 0
        correct = np.zeros(model.num_classes)
        correct_top3 = np.zeros(model.num_classes)
        correct_top5 = np.zeros(model.num_classes)
        correct_top10 = np.zeros(model.num_classes)
        
        count_label = np.zeros(model.num_classes)  # only for micro accuracy
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            loss, output = loss_object.compute_loss(config, model, data, target, geo_model=geo_model, bin_distr=bin_distr, satellite_model=satellite_model, satellite=satellite)
            
            target['label'] = target['label'].to(device)
            validation_loss += loss.item()
            
            label = target['label'].cpu()

            for i in range(len(label)):
                l = label[i].item()
                if l not in predictions_per_obs:
                    predictions_per_obs[l] = []
                predictions_per_obs[l].append(output[i].cpu().detach().numpy())

            #count_label[label] += 1

            for k in [3,5,10]:
                prob_pred_topk, class_pred_topk = torch.topk(output, k, dim=1,
                                                         largest=True,
                                                         sorted=True,
                                                         out=None)
                is_top_k = class_pred_topk.eq(
                    target['label'].view(-1, 1)).sum(dim=-1)
                is_top_k = np.squeeze(is_top_k.cpu().numpy())
                if k == 3:
                    #correct_top3[label] += is_top_k
                    class_pred_top3 = class_pred_topk
                elif k == 5:
                    #correct_top5[label] += is_top_k
                    class_pred_top5 = class_pred_topk
                elif k == 10:
                    #correct_top10[label] += is_top_k
                    class_pred_top10 = class_pred_topk
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(len(label)):
                curr_label = label[i].item()
                if curr_label not in results:
                    results[curr_label] = [0, 0, 0, 0, 0]
                results[curr_label][0] += 1
                count_label[curr_label] += 1
                if pred[i] == curr_label:
                    results[curr_label][1] += 1
                    correct[curr_label] += 1
                if curr_label in class_pred_top3[i]:
                    correct_top3[curr_label] += 1
                    results[curr_label][2] += 1
                if curr_label in class_pred_top5[i]:
                    correct_top5[curr_label] += 1
                    results[curr_label][3] += 1
                if curr_label in class_pred_top10[i]:
                    correct_top10[curr_label] += 1
                    results[curr_label][4] += 1

            is_correct = pred.eq(target['label'].view_as(pred))
            is_correct = np.squeeze(is_correct.cpu().numpy())
            
            #correct[label] += is_correct

        val_loss = validation_loss/len(data_loader)
        top1_acc = np.sum(correct) / np.sum(count_label)
        top3_acc = np.sum(correct_top3) / np.sum(count_label)
        top5_acc = np.sum(correct_top5) / np.sum(count_label)
        top10_acc = np.sum(correct_top10) / np.sum(count_label)
        correct = correct[count_label > 0]
        correct_top3 = correct_top3[count_label > 0]
        correct_top5 = correct_top5[count_label > 0]
        correct_top10 = correct_top10[count_label > 0]
        count_label = count_label[count_label > 0]
        micro_top1_acc = np.mean(correct / count_label)
        micro_top3_acc = np.mean(correct_top3/count_label)
        micro_top5_acc = np.mean(correct_top5/count_label)
        micro_top10_acc = np.mean(correct_top10/count_label)
        
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
        print('epoch : {} {}_loss : {:.4e} | top1_acc {:.4%} (macro) /  '
              '{:.4%} (micro) | top3_acc {:.4%} (macro) '
              '/ {:.4%} (micro) | top5_acc {:.4%} (macro) / {:.4%} (micro) top10_acc {:.4%} (macro) / {:.4%} (micro)'.format(epoch, split, val_loss,
                                        top1_acc, micro_top1_acc,
                                        top3_acc, micro_top3_acc,
                                        top5_acc, micro_top5_acc,
                                        top10_acc, micro_top10_acc))

        experiment.log_metric("{}_loss".format(split), val_loss, step=epoch)
        experiment.log_metric("{}_macro_top1".format(split), top1_acc, step=epoch)
        experiment.log_metric("{}_micro_top1".format(split), micro_top1_acc, step=epoch)
        experiment.log_metric("{}_macro_top3".format(split), top3_acc, step=epoch)
        experiment.log_metric("{}_micro_top3".format(split), micro_top3_acc, step=epoch)
        experiment.log_metric("{}_macro_top5".format(split), top3_acc, step=epoch)
        experiment.log_metric("{}_micro_top5".format(split), micro_top3_acc, step=epoch)
        experiment.log_metric("{}_macro_top10".format(split), top3_acc, step=epoch)
        experiment.log_metric("{}_micro_top10".format(split), micro_top3_acc, step=epoch)

        loss_object.on_epoch_end()
    return writer, eval_score, loss_object, results


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
            # data, target = data.to(device), (target['label']).to(device)
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
    parser.add_argument('--logfile', required=True,
                        type=str, help="name of log file"
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
    parser.add_argument('--geo_only', action="store_true")
    parser.add_argument('--path_to_geo_bins', default=None)
    parser.add_argument('--elev', action='store_true')
    parser.add_argument('--dates', action='store_true')
    parser.add_argument('--clean_data', action='store_true')
    parser.add_argument('--geo_model', type=str)
    parser.add_argument("--use_imgs", action="store_true")
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
    # Get the parameters accordiing to the configuration
    config_dict['geo_bins'] = args.path_to_geo_bins
    config_dict['geo_only'] = args.geo_only
    config_dict['dates'] = args.dates
    config_dict['elev'] = args.elev
    config_dict['geo_model'] = args.geo_model
    config_dict['clean_data'] = args.clean_data
    config_dict['use_imgs'] = args.use_imgs
    config_dict['num_classes'] = 601
 
    config_dict['dataset']['image_dir'] = "new_test_imgs"
    config_dict['dataset']['test_info_file'] = "new_test_samples.csv"

    print("ARGS:::{}".format(args))
    print("CONFIG:::{}".format(config_dict))
    config = ObjFromDict(config_dict)
    experiment.set_name(args.logfile)

    model = get_model(config.model, config.training.loss.type)
    bin_distr = None
    if config.geo_bins:
        train_dataloader, val_dataloader, bin_distr = setup_dataset(config, debug=args.debug)
    else:
        train_dataloader, val_dataloader = setup_dataset(config, debug=args.debug)
    test_dataloader = setup_dataset(config, debug=args.debug, test_only=True)
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
    if resumed:
        training_state = torch.load(os.path.join(
            log_path, "training_state.pth.tar"), map_location=torch.device('cpu'))

        start_epoch = training_state['epoch']
        model.load_state_dict(training_state['state_dict'])
        optimizer.load_state_dict(training_state['optimizer'])
        lr_scheduler.load_state_dict(training_state['lr_scheduler'])
        best_scores = training_state['best_scores']


    eval_args = [config, model, val_dataloader, loss_object, device,
                 0, "val", writer, bin_distr]
    _, _, _, val_results = evaluate(*eval_args, metrics=metrics)
    eval_args = [config, model, test_dataloader, loss_object, device,
                 0, "test", writer, bin_distr]
    writer, eval_score, loss_object, test_results = evaluate(*eval_args, metrics=metrics)
    lr_scheduler.step()

    writer.close()



if __name__ == '__main__':
    main()
