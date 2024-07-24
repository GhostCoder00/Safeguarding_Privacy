import numpy as np
import csv
import os
import pandas as pd
import torch
import cv2
import scipy.io

"""
In the Korean dataset, one video (~1h) is recorded for each participant. First, we need to cut 
the labeled video into 300-frame videos, which are used for training and testing, based on a 
given log file. Then, we need to save the 300-frame videos into separate folders for each participant. 
"""

df    = pd.read_csv('../Mind_Wandering_Detection_Data_Korea/fold_ids.csv')
participants_fail = {}

part_id = df['participant_id'].values.tolist()
uuid = df['uuid'].values.tolist()
mw_label = df['mw_label'].values.tolist()
start_frame = df['frame'].values.tolist() #exact time of the mind wandering/non mind wandering labels
participants = []

vidpath = '../Mind_Wandering_Detection_Data_Korea/data/' #path to the 1h long videos
outpath = '../Mind_Wandering_Detection_Data_Korea/data_separate_videos/'


for idx in range(len(part_id)):
    if part_id[idx] not in participants:
        participants.append(part_id[idx])
        vidfile = cv2.VideoCapture(vidpath+part_id[idx]+'/'+'recording.mp4')
        if (vidfile.isOpened() == False):
            print("Error opening the video file")
        fps = vidfile.get(cv2.CAP_PROP_FPS)
        frame_count = vidfile.get(cv2.CAP_PROP_FRAME_COUNT)

        frames = []
        #frame extraction
        while(vidfile.isOpened()):
            ret, frame = vidfile.read()
            if ret == True:
                #frame = np.reshape(frame, (3, 480, 640))
                frames.append(frame) 
            else:
                break
        vidfile.release()
        if not os.path.exists(outpath+part_id[idx]):
              os.makedirs(outpath+part_id[idx])
    currect_video_frames = frames[start_frame[idx]-300:start_frame[idx]] #300 frames before the label
    out = cv2.VideoWriter(outpath+part_id[idx]+'/'+uuid[idx]+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480)) #save the 300 frames into a video
    for fr in currect_video_frames:
        out.write(fr) # frame is a numpy.ndarray with shape (1280, 720, 3)
    out.release()



        