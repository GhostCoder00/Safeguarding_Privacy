import torch
import wandb
import tqdm
import copy

# self-defined functions
from models import LSTM, Dummy, model_train, model_eval_bagging
from utils import Args, seed
from client import get_clients

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
def main_bagging(args: object) -> None:
    """
    Main function for ensemble bagging.

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
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = False)
    test_loader   = torch.utils.data.DataLoader(test_dataset , batch_size = args.batch_size, shuffle = False)
    print("\nlength of train dataset:", len(train_dataset))
    print(  "length of test  dataset:", len(test_dataset ))
    
    # model initialization
    if args.same_init:
        model = Dummy(args) if args.dummy else LSTM(args)
        models = [copy.deepcopy(model) for _ in range(len(train_clients))]
    else:
        models = [Dummy(args) if args.dummy else LSTM(args) for _ in range(len(train_clients))]

    # wandb init
    wandb.init(entity = 'mind-wandering', project = args.project + '-bagging', name = args.name + ' ' + models[0].__class__.__name__, config = args.__dict__, anonymous = "allow")
    
    # performance before training
    wandb_log = {}
    # model_eval_bagging(models, train_loader, wandb_log, args.hard_vote, 'train/')
    model_eval_bagging(models, test_loader , wandb_log, args.hard_vote, 'test/' )
    wandb.log(wandb_log, step = 0)
    
    # training loop
    print()
    epochs = int(args.global_epoch / args.client_epoch) + 1 # speed up training
    print('number of speed up epochs:', epochs)
    for current_epoch in tqdm.tqdm(range(epochs)):
        # client local training
        for c, m in zip(train_clients, models):
            _ = c.local_train(m, None, None)
            
        # train and validation metrics
        wandb_log = {}
        # model_eval_bagging(models, train_loader, wandb_log, args.hard_vote, 'train/')
        model_eval_bagging(models, test_loader , wandb_log, args.hard_vote, 'test/' )
        wandb.log(wandb_log, step = (current_epoch + 1) * args.client_epoch)
    
    # wandb.finish()
    
# main function call
if __name__ == '__main__':
    args = Args(FL = False)
    
    # wandb run name
    args.name  = 'seed ' + str(args.seed) + ' ' 
    args.name += 'bagging: '
    args.name += str(args.client_optim).split('.')[-1][:-2]
    args.name += ' ' + str(args.client_lr)
    args.name += ' hard' if args.hard_vote else ' soft'
    args.name += ' same' if args.same_init else ' diff'
    
    main_bagging(args)