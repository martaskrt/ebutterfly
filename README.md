## TRAINING IMAGE MODEL ##


## TRAINING GEO MODEL ##

cd geo_prior/geo_prior

python3 train_geo_net.py --config_file config_iNat.json --output geo_model --epochs 100 --early_stop_patience 20 --date --train_full

--> config_file: same as for image model (mainly used for data organization)

--> output: name of file where model will be stored

--> epochs: number of training epochs

--> early_stop_patience: stop training after this many epochs if loss does not decrease

--> date: include dates

--> train_full: include eButterfly location observations that don't have images

## TESTING IMAGE + GEO MODELS ##

cd geo_prior/geo_prior

python3 test_geo_prior.py --resume_dir img_model_dir --logfile test --geo_model geo_model.pth.tar --date

--> resume_dir: directory where image model is located

--> logfile: name of file where to log output

--> geo_model: geo model

--> date: include dates
