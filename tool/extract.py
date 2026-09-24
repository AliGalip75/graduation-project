import torch
import numpy as np

def fliplr(img):
    """flip a batch of images horizontally"""
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long().to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature(model, dataloaders, device, ms=[1]):
    features = []
    labels = []
    
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        
        ff = None
        
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            
            for scale in ms:
                if scale != 1:
                    img_scaled = torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
                else:
                    img_scaled = img
                
                img_scaled = img_scaled.to(device)
                outputs = model(img_scaled)
                
                if ff is None:
                    ff = torch.zeros_like(outputs)
                ff += outputs
                
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        
        features.append(ff.cpu())
        labels.extend(label.numpy())
        
    features = torch.cat(features, 0)
    labels = np.array(labels)
    
    return features, labels
