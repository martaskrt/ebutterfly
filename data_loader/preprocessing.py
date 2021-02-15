import cv2
import numpy as np


class Preprocessing(): 
    def __init__(self, config): 
        self.transforms = []
        if hasattr(config, 'resize'):
            self.transforms.append(Resize(config.resize.scale, keep_ratio=config.resize.keep_ratio))
        if hasattr(config, 'random_rotation'): 
            self.transforms.append(Random_rotation(config.random_rotation.angle))
        if hasattr(config, 'random_crop'): 
            self.transforms.append(Random_crop(config.random_crop.proportion))
        if hasattr(config, 'random_flip'): 
            self.transforms.append(Random_flip(config.random_flip.probability))
            
    def __call__(self,img): 
        for t in self.transforms: 
            img = t(img)
        return img
    
    
class Resize():
    
    def __init__(self, scale, keep_ratio=True):
            self.scale = scale
            self.keep_ratio = keep_ratio
            
    def __call__(self, img): 
        if self.keep_ratio:
            h, w, _ = img.shape
            ratio = max(self.scale[0] / h, self.scale[1] / w)
            img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)
            img = cv2.copyMakeBorder(img, max(0, self.scale[0] - img.shape[0]) // 2, 
                                         max(0, self.scale[0] - img.shape[0]) // 2 + 1,
                                         max(0, self.scale[1] - img.shape[1]) // 2,
                                         max(0, self.scale[1] - img.shape[1]) // 2 + 1,
                                         cv2.BORDER_CONSTANT, 0)
            h, w, _ = img.shape
            img = img[h // 2 - self.scale[0] // 2 : h // 2 + self.scale[0] // 2 ,
              w // 2 - self.scale[1] // 2 : w // 2 + self.scale[1] // 2 ,:]
        else: 
            raise NotImplementedError('keep ratio == False is  not implemented')
        return img
    
    
class Random_rotation():
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, img):
        rows,cols,_ = img.shape
        angle = (2 * np.random.random() - 1) * self.angle 
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angle,1)
        img = cv2.warpAffine(img,M,(cols,rows))
        return img
        
class Random_crop():
    def __init__(self, proportion):
        self.proportion = proportion
    def __call__(self, img): 
        proportion = 1 - (1- self.proportion) * np.random.random()
        h, w, _ = img.shape 
        x_min, y_min = np.random.randint(0, max(1,h - proportion * h)) , np.random.randint(0, max(1,w - proportion * w)) 
        img = img[x_min : x_min + int(proportion * h),
                  y_min : y_min + int( proportion * w), :]
        return img
    
class Random_flip():
    def __init__(self, probability):
        self.probability = probability
    def __call__(self, img): 
        if np.random.random() < self.probability :
            img = cv2.flip(img, flipCode=1)
        return img