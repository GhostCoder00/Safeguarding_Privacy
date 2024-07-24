import numpy as np
import torch
import wandb
import tqdm
import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from utils import get_weight_vector, get_weight_dict
from data_preprocessing import data_normalization

"""
This file contains the model, dataloader, dataset, and training loop for glasses detection.
The resulting model is saved as meglass_resnet18.pth.
"""

class DatasetG(torch.utils.data.Dataset):
    """
    Dataset class for glasses detection.
    """
    def __init__(self, labels: torch.Tensor, data: torch.Tensor, global_weight_dict: dict = None)-> None:
        """
        Arguments:
            labels {torch.Tensor} -- labels of the dataset
            data {torch.Tensor} -- features of the dataset
            global_weight_dict {dict} -- dictionary of weights for each class
        """

        self.labels       = labels
        self.data         = data
        if global_weight_dict is None:
            self.weight_dict = get_weight_dict(self.labels)
        else:
            self.weight_dict = global_weight_dict
        self.weights = get_weight_vector(self.weight_dict, labels)
        self.data = data_normalization(self.data)
        
    def __len__(self)-> int:
        return len(self.labels)

    def __getitem__(self, idx: int)-> dict:
        label   = self.labels [idx]
        weight  = self.weights[idx]
        feature = self.data   [idx]   
        return {'label': label,
                'data': feature, 
                'weight': weight}

class GlassesModel(nn.Module):
    """
    Model for glasses detection.
    """
    def __init__(self)-> None:
        super().__init__()

        # loading a pretrained ResNet18 model
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

        in_features_b7 = 1000

        #for saving the feature vectors before the last linear layer    
        self.glasses = nn.Sequential(
                       nn.BatchNorm1d(in_features_b7),
                       nn.Linear(in_features_b7, 512),
                       nn.ReLU(),
                       nn.BatchNorm1d(512),
                       nn.Linear(512, 256),
                       nn.ReLU()
                       )
        # last linear layer for binary glass detection
        self.glasses_final = nn.Sequential(
                       nn.BatchNorm1d(num_features=256),
                       nn.Dropout(0.4),
                       nn.Linear(256, 1),
                       nn.Sigmoid()
                       )
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.model(x)          # shape of x will be: (N,2560,1,1)                   
        x = torch.flatten(x, start_dim=1)

        g_features = self.glasses(x)
        pred = self.glasses_final(g_features)
    
        return pred, g_features
    
    @property
    def save(self, path: str):
        print('Saving model... %s' % path)
        torch.save(self, path)

################ SETTINGS #########################
batch_size = 256
num_epochs = 30
lr = 1e-4
fone_weight = 'weighted'
###################################################

#datasets
imagenames = []
labels = []
datapath = '../MeGlass_120x120/' #path to the dataset

#read the labels and image names
with open(datapath+'labels.txt') as f:
    lines = f.readlines()
    for line in lines:
        imagenames.append(line.split(' ')[0])
        labels.append(int(line.split(' ')[1][0]))

# divide the dataset into train, validation and test sets
random_state_par = 6
train_val_set_names, test_set_names = train_test_split(imagenames, train_size=0.90, random_state=random_state_par)
train_set_names, val_set_names = train_test_split(train_val_set_names, train_size=0.88, random_state=random_state_par)

# data preprocessing
train_data = torch.zeros(len(train_set_names), 3, 120, 120)
val_data = torch.zeros(len(val_set_names), 3, 120, 120)
test_data = torch.zeros(len(test_set_names), 3, 120, 120)
train_label = torch.zeros(len(train_set_names))
val_label = torch.zeros(len(val_set_names))
test_label = torch.zeros(len(test_set_names))
for i, img_name in enumerate(imagenames):
        path2im = datapath + img_name
        img = np.asarray(Image.open(path2im))
        img = np.reshape(img, (3, 120, 120))
        img_idx = imagenames.index(img_name)
        if img_name in train_set_names:
            train_idx = train_set_names.index(img_name)
            train_data[train_idx] = torch.tensor(img)
            train_label[train_idx] = labels[img_idx]
        elif img_name in val_set_names:
            val_idx = val_set_names.index(img_name)
            val_data[val_idx] = torch.tensor(img)
            val_label[val_idx] = labels[img_idx]
        elif img_name in test_set_names:
            test_idx = test_set_names.index(img_name)
            test_data[test_idx] = torch.tensor(img)
            test_label[test_idx] = labels[img_idx]

#datasets
train_set = DatasetG(torch.Tensor(train_label), torch.Tensor(train_data))
val_set = DatasetG(torch.Tensor(val_label), torch.Tensor(val_data))
test_set = DatasetG(torch.Tensor(test_label), torch.Tensor(test_data))

#dataloaders
train_dataloader  = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, drop_last=True)
val_dataloader  = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True, drop_last=True)
test_dataloader   = torch.utils.data.DataLoader(test_set , batch_size = 1, shuffle = True, drop_last=True)

date_of_t = datetime.datetime.now()


wandb.init(name=str(date_of_t)[5:10] + '_' + str(num_epochs)+'lr'+str(lr), project='Glasses_Detection')
wandb.define_metric("step")
wandb.define_metric("train/*", step_metric="step")
wandb.define_metric("val/*", step_metric="step")

optim_args = {"lr": lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GlassesModel()
model.to(device)
optim = torch.optim.Adam(model.parameters(), **optim_args)

#training loop
for epoch in tqdm.tqdm(range(num_epochs)):
    #train
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_fone = 0.0
    epoch_mcc = 0.0
    for batch in train_dataloader:
        inputs = batch['data'].to(device)
        target_labels = batch['label']
        optim.zero_grad()
        preds, feats = model(inputs)
        bce_loss = nn.BCELoss(reduction='mean') #, weight=batch['weight'].to(device)
        loss = bce_loss(torch.squeeze(preds.to(device)), target_labels.to(device))
        loss.backward()
        
        optim.step()
        epoch_loss += loss.detach().cpu()
        epoch_acc += accuracy_score(y_true=batch['label'].numpy(), y_pred=torch.squeeze(torch.round(preds)).detach().cpu().numpy())
        epoch_fone += f1_score(y_true=batch['label'].numpy(), y_pred=torch.squeeze(torch.round(preds)).detach().cpu().numpy(), average=fone_weight)
        epoch_mcc += matthews_corrcoef(y_true=batch['label'].numpy(), y_pred=torch.squeeze(torch.round(preds)).detach().cpu().numpy())
    
    wandb.log({"step": epoch,
            "train/loss": epoch_loss/(len(train_dataloader)),
            "train/acc": epoch_acc/(len(train_dataloader)),
            "train/f1": epoch_fone/(len(train_dataloader)),
            "train/mcc": epoch_mcc/(len(train_dataloader))})
    
    #validation
    model.eval()
    with torch.no_grad():
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        val_epoch_fone = 0.0
        val_epoch_mcc = 0.0
        for batch in val_dataloader:
            inputs = batch['data'].to(device)
            target_labels = batch['label'].to(device)
            preds, feats = model(inputs)
            bce_loss = nn.BCELoss(reduction='mean')
            loss = bce_loss(torch.squeeze(preds.to(device)), target_labels)

            val_epoch_loss += loss.detach().cpu()
            val_epoch_acc += accuracy_score(y_true=target_labels.cpu().numpy(), y_pred=torch.squeeze(torch.round(preds)).detach().cpu().numpy())
            val_epoch_fone += f1_score(y_true=target_labels.cpu().numpy(), y_pred=torch.squeeze(torch.round(preds)).detach().cpu().numpy(), average=fone_weight)
            val_epoch_mcc += matthews_corrcoef(y_true=batch['label'].numpy(), y_pred=torch.squeeze(torch.round(preds)).detach().cpu().numpy())
        
        wandb.log({"step": epoch,
                "val/loss": val_epoch_loss/(len(val_dataloader)),
                "val/acc": val_epoch_acc/(len(val_dataloader)),
                "val/f1": val_epoch_fone/(len(val_dataloader)),
                "val/mcc": val_epoch_mcc/(len(val_dataloader))})

        avg_val_loss = val_epoch_loss/(len(val_dataloader))

#test
model.eval()
with torch.no_grad():
    test_epoch_loss = 0.0
    test_epoch_acc = 0.0
    test_epoch_fone = 0.0
    gt = []
    preds = []
    for batch in test_dataloader:
        inputs = batch['data'].to(device)
        gt.append(batch['label'].numpy()) 
        pred, _ = model(inputs)
        preds.append(torch.squeeze(torch.round(pred)).detach().cpu().numpy())

        test_epoch_acc += accuracy_score(y_true=np.asarray(gt), y_pred=np.asarray(preds))
        test_epoch_fone += f1_score(y_true=np.asarray(gt), y_pred=np.asarray(preds), average=fone_weight)
    
    wandb.log({"test/acc": test_epoch_acc/(len(test_dataloader.sampler)),
            "test/f1": test_epoch_fone/(len(test_dataloader.sampler))})

torch.save(model.state_dict(), './resnet18'+str(lr)+'.pth') #save the model
wandb.finish()