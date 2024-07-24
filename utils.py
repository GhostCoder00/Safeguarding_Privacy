import torch
import numpy as np
import copy
import random
import argparse
from datetime import datetime

def seed(seed: int) -> None:
    """
    Set random seed.

    Arguments:
        seed (int): random seed.
    """

    print('\nrandom seed:', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def Args(FL: bool) -> argparse.Namespace:
    """
    Helper function for argument parsing.

    Arguments:
        FL (bool): whether to run federated learning or centralized learning.

    Returns:
        args (argparse.Namespace): parsed argument object.
    """

    parser = argparse.ArgumentParser()
    
    # general parameters for both non-FL and FL
    parser.add_argument('-p', '--project', type = str, default = 'korea', help = 'project name, from colorado, korea, daisee, engagenet')
    parser.add_argument('--name', type = str, default = 'name', help = 'wandb run name')
    parser.add_argument('-seed', '--seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('-fl_csv', '--fl_csv', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use FL data split or non-FL data split')
    parser.add_argument('-mg', '--use_meglass', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether to use meglass feature')
    parser.add_argument('--h_size', type = int, default = 100, help = 'hidden size of LSTM model')
    parser.add_argument('--n_layer', type = int, default = 3, help = 'hidden layers of LSTM model')
    parser.add_argument('-sq', '--seq_length', type = int, default = 124, help = 'data sequence length')
    parser.add_argument('-bs', '--batch_size', type = int, default = 4, help = 'batch size')
    parser.add_argument('-bd', '--bi_dir', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use bidirectional LSTM')
    parser.add_argument('-c_lr', '--client_lr', type = float, default = -3, help = 'client learning rate in exponent')
    parser.add_argument('--global_epoch', type = int, default = 401, help = 'number of global aggregation rounds')
    parser.add_argument('-c_op', '--client_optim', default = torch.optim.SGD, help = 'client optimizer')
                    
    # general parameters for FL
    parser.add_argument('-fl', '--switch_FL', type = str, default = 'FedAvg', help = 'FL algorithm, from FedAvg, FedAdam, FedProx, MOON, FedAwS, TurboSVM')
    parser.add_argument('-C', '--client_C', type = float, default = 0.5, help = 'number of participating clients in each aggregation round')
    parser.add_argument('-E', '--client_epoch', type = int, default = 8, help = 'number of client local training epochs')
    
    # for FedOpt (FedAdam)
    parser.add_argument('-g_lr', '--global_lr', type = float, default = -3, help = 'global learning rate in exponent')
    parser.add_argument('-g_op', '--global_optim', default = torch.optim.Adam, help = 'global optimizer')
    
    # for FedAwS and TurboSVM
    parser.add_argument('-l_lr', '--logits_lr', type = float, default = -3, help = 'global learning rate for logit layer in exponent')
    parser.add_argument('-l_op', '--logits_optim', default = torch.optim.Adam, help = 'global optimizer for logit layer')

    # dummy classifier
    parser.add_argument('-dummy', '--dummy', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether to use dummy classifier')
    
    args = parser.parse_args()
    args.time = str(datetime.now())[5:-10]
    args.fed_agg = None
    args.MOON = False
    args.FedProx = False

    # reuse optimizer or not
    args.FL = FL
    args.reuse_optim = not FL

    # features
    args.which_feature = ['emonet', 'openface_8']
    if args.use_meglass:
        args.which_feature.append('meglass')
    if args.dummy:
        args.which_feature.append('dummy')

    # paths
    args.paths = {
            'colorado': {
                        'data_split_csv_path_FL' : './datasets/colorado/dataset_summary_colorado_random_FL.csv',
                        'data_split_csv_path_nFL': './datasets/colorado/dataset_summary_colorado_random.csv'   , 
                        'data_csv_path'          : './datasets/colorado/fold_ids_reduced.csv'                  ,
                        'emonet_path'            : '../colorado/emonet/'                     ,
                        'openface_path'          : '../colorado/openface/'                   ,
                        'meglass_path'           : '../colorado/emonet/'                     ,
                        },
            'korea'   : {
                        'data_split_csv_path_FL' : './datasets/korea/dataset_summary_korea_random_FL.csv',
                        'data_split_csv_path_nFL': './datasets/korea/dataset_summary_korea_random.csv'   ,
                        'data_csv_path'          : './datasets/korea/fold_ids.csv'                       ,
                        'emonet_path'            : '../korea/emonet/'                  ,
                        'openface_path'          : '../korea/openface/'                ,
                        'meglass_path'           : '../korea/meglass/'                 ,
                        },
            'daisee'  : {
                        'data_split_csv_path_FL' : './datasets/daisee/dataset_summary_daisee_boredom.csv',
                        'data_split_csv_path_nFL': './datasets/daisee/dataset_summary_daisee_boredom.csv',
                        'data_csv_path'          : './datasets/daisee/fold_ids.csv'                      ,
                        'emonet_path'            : '../daisee/emonet/'                 ,
                        'openface_path'          : '../daisee/openface/'               ,
                        'meglass_path'           : '../daisee/meglass/'                ,
                        },
            'engagenet':{
                        'data_split_csv_path_FL' : './datasets/engagenet/dataset_summary_engagenet.csv',
                        'data_split_csv_path_nFL': './datasets/engagenet/dataset_summary_engagenet.csv',
                        'data_csv_path'          : './datasets/engagenet/fold_ids.csv'                 ,
                        'emonet_path'            : '../engagenet/emonet/'            ,
                        'openface_path'          : '../engagenet/openface/'          ,
                        'meglass_path'           : '../engagenet/meglass/'           ,
                        }
        }

    return args
        
def switch_FL(args: argparse.Namespace) -> None:
    """
    Set hyperparameters according to the choice of federated learning algorithm.

    Arguments:
        args (argparse.Namespace): parsed argument object.
    """

    match args.switch_FL:

        case 'FedAvg':
            args.fed_agg = 'FedAvg'

        case 'FedAdam':
            args.fed_agg = 'FedOpt'
            
        case 'FedProx':
            args.fed_agg = 'FedAvg'
            args.FedProx = True

        case 'MOON':
            args.fed_agg = 'FedAvg'
            args.MOON = True

        case 'FedAwS':
            args.fed_agg = 'FedAwS'
            
        case 'TurboSVM':
            args.fed_agg = 'TurboSVM'
            
        case _:
            raise Exception("wrong switch_FL:", args.switch_FL)

def get_imbalance_weight(labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate class imbalance and assign a weight to each class.

    Arguments:
        labels (torch.Tensor): all sample labels.

    Returns:
        weight (torch.Tensor): weight per class.
    """
    unique_labels, unique_counts = labels.unique(return_counts = True, sorted = True)
    assert(unique_labels.equal(torch.tensor([0., 1.])))
    sum_counts = sum(unique_counts)
    weight = torch.tensor([unique_counts[1] / sum_counts, unique_counts[0] / sum_counts])
    return weight

def weighted_avg_params(params: list[dict[str, torch.Tensor]], weights: list[int] = None) -> dict:
    """
    Compute weighted average of client models.

    Argument:
        params (list[dict[str, torch.Tensor]]): client model parameters. Each element in this list is the state_dict of a client model.
        weights (list[int]): weight per client. Each element in this list is the number of samples of a client.

    Returns:
        params_avg (dict): averaged global model parameters (state_dict), which can be loaded using global_model.load_state_dict.
    """

    if weights == None:
        weights = [1.0] * len(params)
        
    params_avg = copy.deepcopy(params[0])
    for key in params_avg.keys():
        params_avg[key] *= weights[0]
        for i in range(1, len(params)):
            params_avg[key] += params[i][key] * weights[i]
        params_avg[key] = torch.div(params_avg[key], sum(weights))
    return params_avg

def weighted_avg(values: any, weights: any) -> any:
    """
    Calculate weighted average of a vector of values.

    Arguments:
        values (any): values. Can be list, torch.Tensor, numpy.ndarray, etc.
        weights (any): weights. Can be list, torch.Tensor, numpy.ndarray, etc.

    Returns:
        any: weighted average value.
    """

    sum_values = 0
    for v, w in zip(values, weights):
        sum_values += v *w
    return sum_values / sum(weights)

def weight_to_vec(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Assign weight to each sample.

    Arguments:
        w (torch.Tensor): weight per class.
        y (torch.Tensor): all sample labels.

    Returns:
        wv (torch.Tensor): weight per sample.
    """

    w = w.to(y.device)
    wv = w[y.long()]
    return wv