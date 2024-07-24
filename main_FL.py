import torch
import wandb

# self-defined functions
from models import LSTM, Dummy
from client import get_clients
from server import federated_learning
from utils import Args, seed, switch_FL

def main_FL(args: object) -> None:
    """
    Helper function for federated learning. It calls federated_learning in server.py in runtime.

    Arguments:
        args (argparse.Namespace): parsed argument object. 
    """
    # some print
    print("\nusing device:", 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # reproducibility
    seed(args.seed)

    # get train clients and test clients
    train_clients, test_clients = get_clients(args)

    # model initialization
    global_model = Dummy(args) if args.dummy else LSTM(args)

    # wandb init
    wandb.init(project = args.project, name = args.name + ' ' + global_model.__class__.__name__, config = args.__dict__, anonymous = "allow")
    
    # federated learning
    federated_learning(args, train_clients, test_clients, global_model)
    
# main function call
if __name__ == '__main__':
    args = Args(FL = True)
    
    # switch FL algorithm
    switch_FL(args)

    # wandb run name
    args.name  = 'seed ' + str(args.seed) + ' '
    args.name += args.switch_FL + ': '
    match args.switch_FL:
        case 'FedAvg' | 'FedProx' | 'MOON':
            args.name += 'c_lr ' + str(args.client_lr)
        case 'FedAdam':
            args.name += 'c_lr ' + str(args.client_lr) + ' g_lr ' + str(args.global_lr)
        case 'FedAwS' | 'TurboSVM':
            args.name += 'c_lr ' + str(args.client_lr) + ' l_lr ' + str(args.logits_lr)
        case _:
            raise Exception("wrong switch_FL:", args.switch_FL)
    
    main_FL(args)