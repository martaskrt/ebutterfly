import magicpath
import train_geo_net as geo_data_prep
import numpy as np
import torch
import pickle
import torch.nn.functional as F

assert magicpath


class ProbClipMin():
    """ clip probability at close to 0 - indeed we add a smaller slope for
    back propagation"""

    def __init__(self, min_value, alpha):
        self.min_value = min_value
        self.alpha = alpha

    def apply(self, p):
        return torch.max(self.alpha * p + self.min_value,
                         p)


class ProbClipMax():
    """ clip probability at close to 1 - indeed we add a smaller slope for
    back propagation"""

    def __init__(self, max_value, alpha):
        self.max_value = max_value
        self.alpha = alpha

    def apply(self, p):
        return torch.min(self.alpha * p + self.max_value - self.alpha,
                         p)


class Cross_Entropy_Loss():
    '''
    Compute the standard cross entropy loss.
    '''

    def __init__(self, device='cpu'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        with open("species_matrix.pickle", 'rb') as fhandle:
            self.species_matrix = np.array(pickle.load(fhandle))
        np.fill_diagonal(self.species_matrix, 0)
#        self.species_matrix /= self.species_matrix.sum(axis=1, keepdims=True)
        self.species_to_ignore = set()
    def compute_loss(self, config, model, data, target, reduce=True, geo_model=None, bin_distr=None, satellite_model=None, satellite=None):
        data = data.to(self.device)
        output = model(data) # defin another loss function; check that if you input all 1s it will return same thing, plug in probabilities
        # run output through softmax, multiply by probabilities, normalize --> compute cross entropy with this
        img_preds = output.cpu().data.numpy()  

        if geo_model or config.geo_bins or satellite_model:
            output = F.softmax(output, 1)
        if geo_model:
            geo_locs = np.array([[target["Longitude"][i], target["Latitude"][i], target['Elevation'][i]] for i in range(len(target['label']))])
            if config.dates:
                dates = np.array([geo_data_prep.transform_date(target['Date Observed'][i]) for i in range(len(target['label']))])
            else:
                dates = None
            params = {'loc_encode': 'encode_cos_sin', 'date_encode': 'encode_cos_sin', 
                      'use_date_feats': config.dates, 'device': 'cuda'}
            geo_feats = geo_data_prep.generate_feats(config, geo_locs, dates, params=params)
            if config.use_imgs:
                imgs = geo_data_prep.preprocess_img(target["Satellite_imgs"])
            else:
                imgs = []
            geo_embeds = geo_model(geo_feats, imgs=imgs)
            img_preds = output.cpu().data.numpy()
            geo_preds = geo_embeds.cpu().data.numpy()
            if False:
                output = output.cpu().data.numpy() *(geo_embeds.cpu().data.numpy())
                for i in range(len(geo_embeds)):
                    output[i] += np.matmul(output[i].transpose(), self.species_matrix.transpose()).transpose()
            if config.geo_only:
                output = geo_embeds
            else:
                output = output * geo_embeds
                output = output.cpu().data.numpy()
        elif config.geo_bins:
            geo_bins={'geo_bins': target['bin'].cpu(), 'bin_distr': bin_distr}
            for i in range(len(geo_bins['geo_bins'])):
                curr_bin = geo_bins['geo_bins'][i].item()
                if config.geo_only:
                    output[i] = torch.FloatTensor(geo_bins['bin_distr'][curr_bin])
                else:
                    output[i] = (output[i].cpu() * geo_bins['bin_distr'][curr_bin])
                    output[i] = output[i]/output[i].sum() 

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if geo_model:
            output = torch.from_numpy(output).to(device)
        else:
            output = output.to(device)
        target = (target['label']).to(self.device)
        if geo_model or config.geo_bins:
            loss = F.nll_loss(output, target, reduce=reduce)
        else:
            loss = F.cross_entropy(output, target, reduce=reduce)
        return loss, output

    def on_epoch_end(self):
        pass


class Guided_Attention_Loss():
    '''
    Compute the standard cross entropy loss.
    '''

    def __init__(self, device='cuda:0'):
        self.device = device

    def compute_loss(self, model, data, target, reduce=True):
        data, label  = data.to(self.device), (target['label']).to(self.device)
        mask = target['mask'].float().to(self.device)
        output, attention_masks = model(data)
        loss = 0
        for attention_mask in attention_masks:
            _, _, h, w = attention_mask.shape
            kernel_size = (mask.shape[1] // h, mask.shape[2] // w)
            mask_ = torch.nn.functional.max_pool2d(mask,kernel_size)
            mask_ = torch.unsqueeze(mask_,1)
            loss += torch.sum(torch.abs(attention_mask * (1 - mask_))) / torch.sum(torch.abs(attention_mask))
        loss /= len(attention_masks)
        loss += F.cross_entropy(output, label, reduce=reduce)
        return loss, output

    def on_epoch_end(self):
        pass


class Hierarchical_Cross_Entropy_Loss():
    '''
    Computes the hierarchical cross entropy loss.
    '''

    def __init__(self, batch_size, num_classes,
                 weights=None, logarithmic=None,
                 probability_clip=None, tan=None,
                 device='cuda:0'):
        self.device = device
        self.weights = weights
        self.logarithmic = logarithmic
        self.tangente = tan
        if tan is not None:
            self.clip_min = ProbClipMin(**tan['clip_min_params'])
            self.clip_max = ProbClipMax(**tan['clip_max_params'])
        if probability_clip is not None:
            self.probability_clip = ProbClipMin(**probability_clip['params'])
        else:
            self.probability_clip = None
        self.species_losses = []
        self.genus_losses = []
        self.family_losses = []
        self.species_probas = []
        self.genus_probas = []
        self.family_probas = []

    def compute_loss(self, model, data, target, reduce=True):
        data, label = data.to(self.device), (target['label']).to(self.device)
        output = model(data)
        same_family = target["same_family"].to(self.device)
        same_genus = target["same_genus"].to(self.device)
        family_exp = torch.sum(torch.exp(output) * same_family, axis=-1)
        normalization = torch.sum(torch.exp(output), axis=-1)

        family_proba = family_exp / normalization

        genus_exp = torch.sum(torch.exp(output) * same_genus, axis=-1)
        genus_proba = genus_exp / normalization
        species_proba = torch.exp(output[torch.arange(label.shape[0]),
                                         label])
        species_proba = species_proba / normalization
        if self.probability_clip:
            family_proba = self.probability_clip.apply(family_proba)
            genus_proba = self.probability_clip.apply(genus_proba)
            species_proba = self.probability_clip.apply(species_proba)
        if self.logarithmic:
            try:
                k_factor = self.weights['k_factor']
            except TypeError:
                k_factor = 1
            loss = -torch.exp(k_factor * family_proba) * \
                torch.log(family_proba)
            loss += -torch.exp(k_factor * genus_proba) * \
                torch.log(genus_proba)
            loss += -torch.exp(k_factor * species_proba) * \
                torch.log(species_proba)
        elif self.tangente:
            family_proba = self.clip_min.apply(family_proba)
            genus_proba = self.clip_min.apply(genus_proba)
            species_proba = self.clip_min.apply(species_proba)
            family_proba = self.clip_max.apply(family_proba)
            genus_proba = self.clip_max.apply(genus_proba)
            species_proba = self.clip_max.apply(species_proba)
            family_loss = - torch.tan(np.pi/2 * (2*family_proba-1))
            genus_loss = - torch.tan(np.pi/2 * (2*genus_proba-1))
            species_loss = - torch.tan(np.pi/2 * (2*species_proba-1))
            loss = family_loss + genus_loss + species_loss
        else:
            family_loss = - self.weights['Family'] * torch.log(family_proba)
            genus_loss = - self.weights['Genus'] * torch.log(genus_proba)
            species_loss = - self.weights['Species'] * torch.log(species_proba)
            loss = family_loss + genus_loss + species_loss
        self.species_losses += species_loss.detach().cpu().numpy().tolist()
        self.genus_losses += genus_loss.detach().cpu().numpy().tolist()
        self.family_losses += family_loss.detach().cpu().numpy().tolist()
        self.species_probas += species_proba.detach().cpu().numpy().tolist()
        self.genus_probas += genus_proba.detach().cpu().numpy().tolist()
        self.family_probas += family_proba.detach().cpu().numpy().tolist()
        if reduce:
            loss = loss.mean()
        return loss, output

    def on_epoch_end(self):
        print('species loss', np.mean(self.species_losses))
        print('genus loss', np.mean(self.genus_losses))
        print('family losses', np.mean(self.family_losses))
        print('species probas', np.mean(self.species_probas))
        print('genus probas', np.mean(self.genus_probas))
        print('family probas', np.mean(self.family_probas))
        self.species_losses = []
        self.genus_losses = []
        self.family_losses = []
        self.species_probas = []
        self.genus_probas = []
        self.family_probas = []
