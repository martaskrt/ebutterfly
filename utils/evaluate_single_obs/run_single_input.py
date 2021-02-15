import cv2
from resnet import Resnet

import argparse
import datetime
import json
import os


import numpy as np
import pandas as pd
import torch
import geo_models

import pickle
    
from calendar import monthrange
import calendar
import math

import torch.nn.functional as F
def resize(img):
    
    scale = [400, 400]
    keep_ratio = True
    if keep_ratio:
        h, w, _ = img.shape
        ratio = max(scale[0] / h, scale[1] / w)
        img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)
        img = cv2.copyMakeBorder(img, max(0, scale[0] - img.shape[0]) // 2, 
                                     max(0, scale[0] - img.shape[0]) // 2 + 1,
                                     max(0, scale[1] - img.shape[1]) // 2,
                                     max(0, scale[1] - img.shape[1]) // 2 + 1,
                                     cv2.BORDER_CONSTANT, 0)
        h, w, _ = img.shape
        img = img[h // 2 - scale[0] // 2 : h // 2 + scale[0] // 2 ,
          w // 2 - scale[1] // 2 : w // 2 + scale[1] // 2 ,:]
    return img

def preprocess_img(img):
    img = resize(img)
    img = np.transpose(img, (2, 0, 1)).copy().astype(np.float32, copy=True) / 255
    return torch.from_numpy(img).unsqueeze(0)

def encode_loc_time(loc_ip, date_ip, concat_dim=1):
    # assumes inputs location and date features are in range -1 to 1
    # location is lon, lat
    feats = torch.cat((torch.sin(math.pi*loc_ip[:,:2]), torch.cos(math.pi*loc_ip[:,:2])), concat_dim)
    feats_date = torch.cat((torch.sin(math.pi*date_ip.unsqueeze(-1)),
                                    torch.cos(math.pi*date_ip.unsqueeze(-1))), concat_dim)
    feats = torch.cat((feats, feats_date), concat_dim)
    return feats


def convert_loc_to_tensor(x, device=None):
    # intput is in lon {-180, 180}, lat {90, -90}
    xt = x.astype(np.float32)
    xt[:,0] /= 180.0 # longitude
    xt[:,1] /= 90.0 # latitude
    xt = torch.from_numpy(xt)
    if device is not None:
        xt = xt.to(device)
    return xt

def transform_date(date):
    year, month, day = date.split("-")
    year, month, day = int(year), int(month)-1, int(day)-1
    month_count = np.cumsum([monthrange(year, ii+1)[1] for ii in range(12)])
    month_count = np.hstack(([0], month_count))
    num_days = 365.0 if calendar.isleap(year) else 364.0
    dt = (month_count[month] + day) / 365.0
    return round(dt,6)

def generate_feats(locs, dates, device):
    x_locs = convert_loc_to_tensor(locs, device)
    x_dates = torch.from_numpy(dates.astype(np.float32)*2 - 1).to(device)
    feats = encode_loc_time(x_locs, x_dates, concat_dim=1)
    return feats

def evaluate(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data) # defin another loss function; check that if you input all 1s it will return same thing, plug in probabilities
    return output

def main(raw_args=None):
    """
    Main function to run the code. Can either take the raw_args
    in argument or get the arguments from the config_file.
    """


    # -----------------------------------------------------------------------------------------
    # Get the parameters accordiing to the configuration
    #df = pd.read_csv("label2species.csv")
    img_model = torch.jit.load("model.jit")
    #img_model.eval()
    device = 'cpu'
    img_width, img_height = 400, 400
    example_input = torch.ones(1, 3, img_width, img_height)

    if False:
        img_model = Resnet(601)
        device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
        device = 'cpu'
        img_model.to(device)
        training_state = torch.load("best_accuracy.pth", map_location=torch.device('cpu'))
        img_model.load_state_dict(training_state)
        # Create jit model
        img_model.eval()
        img_model_jit = torch.jit.trace(img_model, example_input)
        # Save model
        torch.jit.save(img_model_jit, "model.jit") 


    #img_path = "/home/zach/marta/data/ob_with_images/53d1aa38-baa4-42ff-aa91-325bd8305c28.jpg"
    img_path = "../new_test_imgs/6E04979C-6DF2-440E-721E-397118C7C4B3.jpg"
    #lat = 45.1595
    #lon = -64.3595
    #date = "2018-07-28"
    lat = 44.936731
    lon = -65.070654
    date = "2020-11-22"
    img = cv2.imread(img_path)[:, :, ::-1]
    img = preprocess_img(img)


    num_inputs = 6
    geo_model = torch.jit.load("geo_model.jit")
    example_input = torch.ones(1, 1, 6)
    if False:
        geo_model = geo_models.FCNet(num_inputs=num_inputs, num_classes=img_model.num_classes,num_filts=256).to(device)
        training_state = torch.load("geo_model.pth.tar", map_location=torch.device('cpu'))
        geo_model.load_state_dict(training_state['state_dict'])
        # Create jit model
        geo_model.eval()
        geo_model_jit = torch.jit.trace(geo_model, example_input)
        # Save model
        torch.jit.save(geo_model_jit, "geo_model.jit") 



    geo_locs = np.array([[lon, lat]])
    dates = np.array([transform_date(date)])
    geo_feats = generate_feats(geo_locs, dates, device)
        
            
    img_pred = evaluate(img_model, img, device)
    img_pred = F.softmax(img_pred, 1).cpu().detach().numpy()[0]
    geo_pred = evaluate(geo_model, geo_feats, device).cpu().detach().numpy()[0]

    top_img_preds = np.argsort(img_pred)[::-1][:5]
    top_geo_preds = np.argsort(geo_pred)[::-1][:5]

    print(img_pred[top_img_preds])
    print(geo_pred[top_geo_preds])

    output = img_pred * geo_pred
    output /= output.sum()
    sorted_args = np.argsort(output)[::-1]
    top_1 = sorted_args[0]

    df = pd.read_csv("label2species.csv")
    classes = df['class'].values
    species = df['species'].values

    class2species = {classes[i]: species[i] for i in range(len(species))}
    print("SPECIES PREDICTION:::{}".format(class2species[top_1]))

if __name__ == '__main__':
    main()
