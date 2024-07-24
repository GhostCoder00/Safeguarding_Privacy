import numpy as np
import csv
import os
import pandas as pd
import torch
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from emonet_model import EmoNet
from evaluation import evaluate
import scipy.io
import pandas as pd

def emonet_feature_extraction(file: str, vidpath: str, outpath: str) -> None:
  """
    Helper function for extracting features from videos using EmoNet.

    Arguments:
        file (str): video file name.
        vidpath (str): path to the video file.
        outpath (str): path to save the extracted features in a csv format.        
  """
  if '.mp4' in file or '.avi' in file:
    #open video file
    vidfile = cv2.VideoCapture(vidpath+file)
    if (vidfile.isOpened() == False):
      print("Error opening the video file")
    fps = vidfile.get(cv2.CAP_PROP_FPS)
    frame_count = vidfile.get(cv2.CAP_PROP_FRAME_COUNT)

    frames = []
    features = []
    #framewise feature extraction
    while(vidfile.isOpened()):
        ret, frame = vidfile.read()
        if ret == True:
            # reshaping and converting the frame to torch tensor
            frame = cv2.resize(frame, dsize=(480, 640), interpolation=cv2.INTER_CUBIC)
            frame = np.reshape(frame, (3, 480, 640))
            frames.append(torch.tensor(frame.astype(np.float32)))
            frame = torch.tensor(frame.astype(np.float32))
            test_dataloader_no_flip = DataLoader(torch.unsqueeze(frame, 0), batch_size=1, shuffle=False)
            out = evaluate(net, test_dataloader_no_flip, device=device) #out['features'] is the extracted feature with the EmoNet model
            features.append(np.asarray(out['features'].cpu()))
        else:
            break
    vidfile.release()
    
    #save the extracted features into a csv file
    outfile = open(outpath+file[:-4]+'.csv', 'w', newline='')
    writer = csv.writer(outfile)
    for i in features:
        writer.writerow(i)
    outfile.close() 

###################### set dataset name here #############################

"""
  Main code for extracting features from videos using EmoNet corresponding to the four datasets.
"""
dataset_name = 'daisee' # options: 'korea', 'colorado', 'engagenet', 'daisee'

#load pretrained EmoNet model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path2net = '/pretrained/emonet_5.pth'
state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_5.pth')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = EmoNet(n_expression=5).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

if dataset_name == 'engagenet':
  vidpath = '../EngageNet/data/'
  outpath = '../EngageNet/emonet/'
  for path, subdirs, files in os.walk(vidpath):
    for num_of_vid, file in enumerate(os.listdir(vidpath)):
      emonet_feature_extraction(file, vidpath, outpath)
elif dataset_name == 'korea':
  vidpath = '../Mind_Wandering_Detection_Data_Korea/data_separate_videos/'
  outpath = '../Mind_Wandering_Detection_Data_Korea/features/emonet/'
  for path, subdirs, files in os.walk(vidpath):
    for subdir in subdirs:
      for num_of_vid, file in enumerate(os.listdir(vidpath+subdir+'/')):
        emonet_feature_extraction(file, vidpath+subdir+'/', outpath)
elif dataset_name == 'colorado':
  vidpath = '../MW_vids_Sidney/data/'
  outpath = '../MW_vids_Sidney/emonet/'
  for path, subdirs, files in os.walk(vidpath):
    for num_of_vid, file in enumerate(os.listdir(vidpath)):
      emonet_feature_extraction(file, vidpath, outpath)
elif dataset_name == 'daisee':
  vidpath = '../DAiSEE/DataSet_original_all_data/'
  outpath = '../DAiSEE/emonet/'
  for path_set, dir_sets, file_sets in os.walk(vidpath):
    for dir_set in dir_sets:
      for path, subdirs, files in os.walk(vidpath+dir_set+'/'):
        for subdir in subdirs:
          for path1, subdirs1, files1 in os.walk(vidpath+dir_set+'/'+subdir):
            for subdir1 in subdirs1:
              for file in os.listdir(vidpath+dir_set+'/'+subdir+'/'+subdir1+'/'):
                emonet_feature_extraction(file, vidpath+dir_set+'/'+subdir+'/'+subdir1+'/', outpath)
else:
  print('Dataset name is invalid! Please choose from: korea, colorado, engagenet, daisee')
