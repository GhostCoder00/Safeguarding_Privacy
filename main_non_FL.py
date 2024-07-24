import torch
import wandb
import tqdm

# self-defined functions
from models import LSTM, Dummy, model_train, model_eval
from utils import Args, seed
from client import get_clients

# reuse
data_dict = {}
clients, train_clients, valid_clients, test_clients = [], [], [], []

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
def main_non_FL(args: object) -> None:
    """
    Main function for centralized learning.

    Arguments:
        args (argparse.Namespace): parsed argument object.
    """
    # some print
    print("\nusing device:", device)
    
    # reproducibility
    seed(args.seed)

    # get train clients and test clients
    train_clients, test_clients = get_clients(args)

    # dataset and data loader
    train_dataset = torch.utils.data.ConcatDataset([c.dataset for c in train_clients])
    test_dataset  = torch.utils.data.ConcatDataset([c.dataset for c in test_clients ])
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True )
    test_loader   = torch.utils.data.DataLoader(test_dataset , batch_size = args.batch_size, shuffle = False)
    print("\nlength of train dataset:", len(train_dataset))
    print(  "length of test  dataset:", len(test_dataset ))
    
    # model initialization
    model = Dummy(args) if args.dummy else LSTM(args)
    model.to(device)

    # wandb init
    wandb.init(project = args.project, name = args.name + ' ' + model.__class__.__name__, config = args.__dict__, anonymous = "allow")
    
    # performance before training
    wandb_log = {}
    model_eval(model, train_loader, wandb_log, 'train/')
    model_eval(model, test_loader , wandb_log, 'test/' )
    wandb.log(wandb_log)
    
    # training loop
    print()
    for current_epoch in tqdm.tqdm(range(args.global_epoch)):
        # train for 1 epoch
        model_train(model, train_loader, 1)
        
        # train and validation metrics
        wandb_log = {}
        model_eval(model, train_loader, wandb_log, 'train/')
        model_eval(model, test_loader , wandb_log, 'test/' )
        wandb.log(wandb_log)
    
    
    # model.to('cpu')
    # wandb.finish()
    
# main function call
if __name__ == '__main__':
    args = Args(FL = False)

    # wandb run name
    args.name  = 'seed ' + str(args.seed) + ' ' 
    args.name += 'non-FL: '
    args.name += str(args.client_optim).split('.')[-1][:-2]
    args.name += ' ' + str(args.client_lr)
    
    main_non_FL(args)