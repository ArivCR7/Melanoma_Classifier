import torch
import torchvision
import albumentations
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import pretrainedmodels
import numpy as np

class MelanomaDataLoader(Dataset):
    '''Dataloader class'''
    def __init__(self, npy_data, targets, augmentations=None):
        self.npy_data = npy_data
        self.targets = targets
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.npy_data)
    
    def __getitem__(self, idx):
        
        np_img = self.npy_data[idx]
        target = self.targets[idx]
        if self.augmentations:
            augmented = self.augmentations(image=np_img)
            image_data = augmented['image']
        else:
            image_data = torch.from_numpy(np_img)
        print("image data shape: ",image_data.shape)
        image_data = np.transpose(image_data, (2,0,1)).astype(np.float32)
        return {
            'images': torch.tensor(image_data, dtype=torch.float),
            'targets': torch.tensor(target, dtype=torch.long)
        }

class SEResnext50_32x4d(nn.Module):
    '''This is network class'''
    def __init__(self, pretrained='imagenet', wp = None):
        super(SEResnext50_32x4d, self).__init__()
        
        self.base_model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=None)
        #print(self.base_model)
        if pretrained is not None:
            self.base_model.load_state_dict(
            torch.load('../input/pretrained-model-weights-pytorch/se_resnext50_32x4d-a260b3a4.pth')
            )
        '''for params in self.base_model.parameters():
            params.requires_grad = False'''
            
        self.l0 = nn.Linear(2048, 1)
        if wp is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=wp)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, images, targets):
        batch_size = images.shape[0]
        
        x = self.base_model.features(images)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        yhat = torch.sigmoid(self.l0(x))
        #loss = nn.BCEWithLogitsLoss(pos_weight=wp)(yhat, targets.view(-1, 1).type_as(x))
        loss = self.criterion(yhat, targets.view(-1, 1).type_as(x))
        return yhat, loss

def predict(image_path, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_aug = albumentations.Compose([
        albumentations.Resize(128, 128),
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
    ])
    #just for the sake
    test_target = np.array([0])
    test_wp = None
    np_image = cv2.imread(image_path)
    np_image = np.expand_dims(np_image, axis=0)
    print("Input image shape: ",np_image.shape)
    test_data = MelanomaDataLoader(np_image, test_target, augmentations=test_aug)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False) 
    
    test_preds = []
    #tk1 = tqdm(test_loader, total = len(test_loader))
    model = model.to(device)
    with torch.no_grad():
        for batch, data in enumerate(test_loader, 1):
            torch.cuda.empty_cache()
            images, targets = data['images'], data['targets']
            images, targets = images.to(device), targets.to(device)
            out, _ = model(images, targets)
            #test_preds.append(out)
            test_preds.append(out.cpu())
            del images, targets
    predictions = np.vstack(test_preds).ravel()
    return predictions