import torch
import copy
import pandas as pd
from data_preprocessing import DatasetMW, get_data_dict
from models import model_train, model_train_FedProx, model_train_MOON
from utils import get_imbalance_weight

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Client(object):
    """
    Self-defined client object.
    """

    def __init__(self, args: object, client_name: str, client_data_dict: dict[str, torch.Tensor]) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            client_name (str): client name / id.
            client_data_dict (dict[str, torch.Tensor]): a dictionary holding all data of this client, with 'data', 'labels', 'glasses', and 'meglass' as keys.
        """
        
        super(Client, self).__init__()
        self.client_name  = client_name
        self.client_epoch = args.client_epoch
        self.batch_size   = args.batch_size
        
        # for FedProx
        self.FedProx = args.FedProx

        # for MOON
        self.MOON = args.MOON

        # datasets and data loaders
        self.dataset = DatasetMW(client_data_dict['data'], client_data_dict['labels'], client_data_dict['glasses'], client_data_dict['meglass'])
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = not self.MOON, drop_last = True)
    
    def local_train(self, client_model: torch.nn.Module, global_model: torch.nn.Module, previous_feature: torch.Tensor) -> torch.Tensor:
        """
        Client local training.

        Arguments:
            client_model (torch.nn.Module): pytorch model (client local model).
            global_model (torch.nn.Module): pytorch model (global model).
            previous_feature (torch.Tensor): features extracted by client model in last global epoch, useful for MOON.

        Returns:
            last_client_features (torch.Tensor): features extracted by client model in current global epoch.
        """

        client_model.to(device)

        client_features = []
        if self.MOON:
            for current_client_epoch in range(self.client_epoch):
                # client model train
                if (previous_feature != None) and (client_features == []):
                    client_features_tensor = previous_feature
                elif (previous_feature == None) and (client_features == []):
                    client_features_tensor = None
                elif client_features != []:
                    client_features_tensor = torch.zeros((len(client_features), client_features[0].shape[0], client_features[0].shape[1]))
                    for idx, prev in enumerate(client_features):
                        client_features_tensor[idx] = copy.deepcopy(prev.detach())
                    client_features_tensor = client_features_tensor.cuda()
                    
                client_feat = model_train_MOON(client_model, global_model, self.data_loader, client_features_tensor)
                client_features.append(client_feat)
        elif self.FedProx:
            model_train_FedProx(client_model, global_model, self.data_loader, self.client_epoch)
        else:
            model_train(client_model, self.data_loader, self.client_epoch)

        client_model.to('cpu')
        last_client_features = []
        if self.MOON:
            last_client_features = client_features[-1]
        
        return last_client_features

def get_clients(args: object) -> tuple[list[Client]]:
    """
    Read data into dictionary and intialize client objects using the data dictionary.

    Arguments:
        args (argparse.Namespace): parsed argument object.

    Returns:
        train_clients (list[Client]): training clients.
        test_clients (list[Client]): test / validation clients.
    """
    
    # get data dictionary
    data_dict = get_data_dict(args.paths[args.project]['data_split_csv_path_FL' if args.fl_csv else 'data_split_csv_path_nFL'], 
                              args.paths[args.project]['data_csv_path'], 
                              args.paths[args.project]['emonet_path'], 
                              args.paths[args.project]['openface_path'], 
                              args.paths[args.project]['meglass_path'], 
                              args.which_feature, args.seq_length)
    
    # get class weight
    all_labels = torch.cat([v['labels'] for _, v in data_dict.items()])
    args.imba_weight = get_imbalance_weight(all_labels)
    print('class weight:', args.imba_weight)

    # get clients
    clients = []
    for client_name, client_data_dict in data_dict.items():
        client = Client(args, client_name, client_data_dict)
        clients.append(client)

    # data split (clients split)
    clients_cv_folds = [[], [], [], [], [], []] # 6 folds. If not doing 5-fold CV, the folds 0 ~ 4 are used as train data and the fold 5 is test data 
    split_df = pd.read_csv(args.paths[args.project]['data_split_csv_path_FL' if args.fl_csv else 'data_split_csv_path_nFL'])
    participant_ids = split_df['participant_id']
    participant_cvs = split_df['test_fold_num']
    participant_num = split_df['num of videos']
    for name, cv_fold_id, num_samples in zip(participant_ids, participant_cvs, participant_num):
        if cv_fold_id == 'None' or num_samples < args.batch_size:
            continue
        for c in clients:
            if c.client_name == name:
                clients_cv_folds[int(cv_fold_id)].append(c)
    
    # trai-test-split
    test_clients  = clients_cv_folds.pop()
    train_clients = sum(clients_cv_folds, [])
    print("\nnumber of train clients:", len(train_clients))
    print(  "number of test  clients:", len(test_clients ))

    return train_clients, test_clients