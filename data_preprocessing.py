import pandas as pd
import numpy  as np
import torch
import os
from skimage.measure import block_reduce

def get_data_dict(data_split_csv_path: str, 
                  data_csv_path: str, 
                  emonet_path: str, 
                  openface_path: str, 
                  meglass_path: str, 
                  which_feature: list[str] = ['emonet', 'openface_8'], 
                  seq_length: int = 124
                  ) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read data into a dictionary.

    Arguments:
        data_split_csv_path (str): path to data split csv file.
        data_csv_path (str): path to meta data csv file.
        emonet_path (str): path to emonet feature directory.
        openface_path (str): path to openface feature directory.
        meglass_path (str): path to meglass feature directory.
        which_feature (list[str]): a list strings specifying which features should be used.
        seq_length (int): number of frames of each sample.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as key. Each value entry is also a dictionary with 'data', 'labels', 'meglass', 'glasses' as keys.
    """

    use_emonet      = 'emonet'   in which_feature
    use_openface_8  = 'openface_8' in which_feature  # only use 8 eye features from all openface features
    use_openface_14 = 'openface_14' in which_feature # only use 14 eye features from all openface features
    use_openface    = ('openface' in which_feature) or use_openface_8 or use_openface_14
    use_meglass     = 'meglass' in which_feature
    
    df    = pd.read_csv(data_csv_path)
    names = df['participant_id']
    uuids = df['uuid']
    mws   = df['mw_label']

    df_g = pd.read_csv(data_split_csv_path)
    glasses_val = df_g['glasses'].values.tolist()
    part_name = df_g['participant_id'].values.tolist()
    
    # return value
    data_dict = {unique_name: {'data':[], 'labels':[], 'meglass':[], 'glasses':[]} for unique_name in np.unique(names)}

    for name, uuid, mw in zip(names, uuids, mws):
        glasses = glasses_val[part_name.index(name)]
        if '.mp4' in uuid:
            uuid = uuid[:-4]    # remove '.mp4' file extension
        elif '.avi' in uuid:
            uuid = uuid[:-4]
        data = []
        meglass = []

        of_file_path = openface_path + uuid + '.csv'
        em_file_path = emonet_path + uuid + '.csv'
        mg_file_path = meglass_path + uuid + '.csv'
        if os.path.exists(of_file_path):  # only considering successful frames     
            openface_pd = pd.read_csv(of_file_path)
            successful = openface_pd[' success'].values.tolist()[:seq_length]
            failed_idx = []
            for idx, item in enumerate(successful):
                if item == 0:
                    failed_idx.append(idx)
            if failed_idx != []:
                failed_idx.reverse()

            if os.path.exists(em_file_path) and use_emonet:
                emonet_features = pd.read_csv(em_file_path, header=None).to_numpy()
                if emonet_features.shape[0] < seq_length:
                    continue
                if 'colorado' in data_split_csv_path:
                    for frame_idx, frame in enumerate(emonet_features):
                        if frame_idx == 0 and successful[frame_idx] == 0: # never fulfilled
                            first_successful = successful.index(next(filter(lambda x: x!=0, successful)))
                        elif frame_idx < seq_length and successful[frame_idx] == 0 and frame_idx != 0:
                            emonet_features[frame_idx] = last_successful_frame
                        else:
                            last_successful_frame = frame
                if emonet_features.shape[0] >= seq_length:
                    if 'colorado' in data_split_csv_path:
                        emonet_features = emonet_features[:seq_length]
                    elif 'korea' in data_split_csv_path or 'daisee' in data_split_csv_path or 'engagenet' in data_split_csv_path:
                        for failed in failed_idx:
                            emonet_features = np.delete(emonet_features, (failed), axis=0)
                        downsample_idxs    = np.linspace(0, len(emonet_features) - 1, seq_length).astype(int)
                        emonet_features = emonet_features[downsample_idxs]
                    else:
                        raise Exception("wrong csv name:", data_split_csv_path)
                    data.append(emonet_features)
                else:
                    continue
            
            if os.path.exists(mg_file_path) and use_meglass:
                meglass_features = pd.read_csv(mg_file_path, header=None).to_numpy()
                if 'colorado' in data_split_csv_path:
                    for frame_idx, frame in enumerate(emonet_features):
                        if frame_idx == 0 and successful[frame_idx] == 0: # never fulfilled
                            first_successful = successful.index(next(filter(lambda x: x!=0, successful)))
                        elif frame_idx < seq_length and successful[frame_idx] == 0 and frame_idx != 0:
                            meglass_features[frame_idx] = last_successful_frame_g
                        else:
                            last_successful_frame_g = meglass_features[frame_idx]
                if meglass_features.shape[0] >= seq_length:
                    if 'colorado' in data_split_csv_path:
                        meglass_features = meglass_features[:seq_length]
                    elif 'korea' in data_split_csv_path or 'daisee' in data_split_csv_path or 'engagenet' in data_split_csv_path:
                        for failed in failed_idx:
                            meglass_features = np.delete(meglass_features, (failed), axis=0)
                        downsample_idxs    = np.linspace(0, len(meglass_features) - 1, seq_length).astype(int)
                        meglass_features = meglass_features[downsample_idxs]
                    else:
                        raise Exception("wrong csv name:", data_split_csv_path)
                    meglass.append(meglass_features)
                else:
                    continue

            if use_openface:
                file_path = openface_path + uuid + '.csv'
                if not os.path.exists(file_path):
                    continue
                openface_pd = pd.read_csv(file_path)
                successful = openface_pd[' success']
                openface_pd = openface_pd.to_numpy()
                if 'colorado' in data_split_csv_path:
                    openface_pd = openface_pd[:seq_length]
                    for frame_idx, frame in enumerate(openface_pd):
                        if frame_idx < 125:
                            if frame_idx == 0 and successful[frame_idx] == 0: # never fulfilled
                                first_successful = successful.index(next(filter(lambda x: x!=0, successful)))
                            elif successful[frame_idx] == 0 and frame_idx != 0:
                                openface_pd[frame_idx] = last_successful_frame
                            else:
                                last_successful_frame = frame

                openface_features  = np.delete(openface_pd, [0, 1, 2, 3, 4], axis=1)      # drop first five columns as they are not needed
                if use_openface_8:
                    openface_features = openface_features[:, 0:8]      # selecting only gaze features
                elif use_openface_14:
                    openface_features = np.delete(openface_features, range(8, 287), axis=1)
                    openface_features = np.delete(openface_features, range(13, len(openface_features)), axis=1)
                
                
                if openface_features.shape[0] >= seq_length:
                    if 'colorado' in data_split_csv_path:
                        openface_features = openface_features[:seq_length]
                    elif 'korea' in data_split_csv_path or 'daisee' in data_split_csv_path or 'engagenet' in data_split_csv_path:
                        for failed in failed_idx:
                            openface_features = np.delete(openface_features, (failed), axis=0)
                        downsample_idxs   = np.linspace(0, len(openface_features) - 1, seq_length).astype(int)
                        openface_features = openface_features[downsample_idxs]   
                    else:
                        raise Exception("wrong csv name:", data_split_csv_path)
                    data.append(openface_features)
                else:
                    continue
        elif not os.path.exists(of_file_path) and use_openface:
            continue

        if len(data) > 0:
            data_dict[name]['data']  .append(np.concatenate(data, axis = 1))
            data_dict[name]['labels'].append(1.0 if mw == 'MW' else 0.0)
            if use_meglass:
                data_dict[name]['meglass'].append(np.concatenate(meglass, axis = 1))
            data_dict[name]['glasses'].append(1.0 if glasses == 'Y' else 0.0)
    
    if 'dummy' in which_feature:
        #for our baseline classifier, we use the same extracted features from OpenFace and EmoNet, but we averaged every 10 frames to reduce the input dimension
        for _, v in data_dict.items():
            for idx, element in enumerate(v['data']):
                v['data'][idx] = block_reduce(element[4:, :], block_size=(10,1), func=np.mean)

    # to torch tensors
    for _, v in data_dict.items():
        v['data'  ] = torch.tensor(np.array(v['data'])).float() # (df -> np array -> torch tensor) faster than (df -> torch tensor)
        v['labels'] = torch.tensor(v['labels']) #.long()
        v['meglass'] = torch.tensor(np.array(v['meglass'])).float() if use_meglass else None
        v['glasses'] = torch.tensor(v['glasses'] )#.long()

    return data_dict
 
class DatasetMW(torch.utils.data.Dataset):
    """
    Self-defined dataset class.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, g: torch.Tensor, meglass: torch.Tensor) -> None:
        """
        Arguments:
            x (torch.Tensor): concatenated emonet and openface features.
            y (torch.Tensor): ground truth class labels (e.g. mind wandering vs. non mind wandering).
            g (torch.Tensor): ground truth glass labels (e.g. with glass vs. without glass).
            meglass (torch.Tensor): meglass features.
        """

        self.x = x
        self.y = y
        self.g = g
        
        # meglass_feature in form of mean and std
        if meglass is not None:
            self.g_mean = meglass.mean(dim = 1)
            self.g_std  = meglass.std (dim = 1)
        else:
            self.g_mean = torch.zeros(len(y))
            self.g_std  = torch.zeros(len(y))

    def __len__(self) -> int:
        """
        Returns:
            (int): size of dataset.
        """
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """
        Arguments:
            idx (int): index to sample.

        Returns:
            x (torch.Tensor): sample feature.
            y (torch.Tensor): ground truth class label (e.g. mind wandering vs. non mind wandering).
            g (torch.Tensor): ground truth glass label (e.g. with glass vs. without glass).
            g_mean (torch.Tensor): mean of meglass feature.
            g_std (torch.Tensor): std of meglass feature.
        """
        x = self.x[idx] 
        y = self.y[idx]
        g = self.g[idx]
        g_mean = self.g_mean[idx]
        g_std  = self.g_std [idx]
        
        return x, y, g, g_mean, g_std