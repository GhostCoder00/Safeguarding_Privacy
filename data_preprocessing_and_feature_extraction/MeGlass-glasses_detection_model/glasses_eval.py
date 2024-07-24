import numpy as np
import csv
import os
import pandas as pd
import torch
import cv2
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader

"""
This file saves the features extracted from the MeGlass model.
"""

class GlassesModel(nn.Module):
    """
    Model for glasses detection.
    """
    def __init__(self)-> None:
        super().__init__()

        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

        in_features_b7 = 1000
            
        self.glasses = nn.Sequential(
                       nn.BatchNorm1d(in_features_b7),
                       nn.Linear(in_features_b7, 512),
                       nn.ReLU(),
                       nn.BatchNorm1d(512),
                       nn.Linear(512, 256),
                       nn.ReLU()
                       )
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

def meglass_feature_extraction(file: str, vidpath: str, outpath: str) -> None:
  """
  Saves features from the MeGlass model.
  
  Args:
    file (str): name of video file
    vidpath (str): path to video file
    outpath (str): path to save extracted features in a csv format
  """

  if ('.mp4' in file or '.avi' in file) and (not os.path.exists(outpath+file[:-4]+'.csv')):
    vidfile = cv2.VideoCapture(vidpath+'/'+subdir+'/'+subdir1+'/'+file)
    if (vidfile.isOpened() == False):
      print("Error opening the video file")
    fps = vidfile.get(cv2.CAP_PROP_FPS)
    frame_count = vidfile.get(cv2.CAP_PROP_FRAME_COUNT)

    frames = []
    features = []
    while(vidfile.isOpened()):
        ret, frame = vidfile.read()
        if ret == True:
            frame = np.reshape(frame, (3, 480, 640))
            frames.append(torch.tensor(frame.astype(np.float32)))
            frame = torch.tensor(frame.astype(np.float32))
            test_dataloader = DataLoader(torch.unsqueeze(frame, 0), batch_size=1, shuffle=False)
            for batch in test_dataloader:
                inputs = batch.to(device)
                preds, feats = net(inputs) # feature extraction
                feats = torch.squeeze(feats.detach().cpu())
                features.append(np.asarray(feats)) #feature collection
        else:
            break
    vidfile.release()
    # save extracted features in a csv file
    outfile = open(outpath+file[:-4]+'.csv', 'w', newline='')
    writer = csv.writer(outfile)
    for i in features:
        writer.writerow(i)
    outfile.close()

###################### set dataset name here #############################
dataset_name = 'daisee' # options: 'korea', 'colorado', 'engagenet', 'daisee'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path2net = '/MeGlass-glasses_detection_model/meglass_resnet18.pth' # trained model
state_dict = torch.load(str(path2net), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = GlassesModel().to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

# the diferent datasets have different folder structures
if dataset_name == 'engagenet':
  vidpath = '../EngageNet/data/'
  outpath = '../EngageNet/meglass/'
  for path, subdirs, files in os.walk(vidpath):
    for num_of_vid, file in enumerate(os.listdir(vidpath)):
      meglass_feature_extraction(file, vidpath, outpath)
elif dataset_name == 'korea':
  vidpath = '../Mind_Wandering_Detection_Data_Korea/data_separate_videos/'
  outpath = '../Mind_Wandering_Detection_Data_Korea/features/emonet/'
  for path, subdirs, files in os.walk(vidpath):
    for subdir in subdirs:
      for num_of_vid, file in enumerate(os.listdir(vidpath+subdir+'/')):
        meglass_feature_extraction(file, vidpath+subdir+'/', outpath)
elif dataset_name == 'colorado':
  vidpath = '../MW_vids_Sidney/data/'
  outpath = '../MW_vids_Sidney/emonet/'
  for path, subdirs, files in os.walk(vidpath):
    for num_of_vid, file in enumerate(os.listdir(vidpath)):
      meglass_feature_extraction(file, vidpath, outpath)
elif dataset_name == 'daisee':
  vidpath = '../DAiSEE/DataSet_original_all_data/'
  outpath = '../DAiSEE/meglass/'
  for path_set, dir_sets, file_sets in os.walk(vidpath):
    for dir_set in dir_sets:
      for path, subdirs, files in os.walk(vidpath+dir_set+'/'):
        for subdir in subdirs:
          for path1, subdirs1, files1 in os.walk(vidpath+dir_set+'/'+subdir):
            for subdir1 in subdirs1:
              for file in os.listdir(vidpath+dir_set+'/'+subdir+'/'+subdir1+'/'):
                meglass_feature_extraction(file, vidpath+dir_set+'/'+subdir+'/'+subdir1+'/', outpath)
