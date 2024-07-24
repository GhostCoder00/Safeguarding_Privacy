import torch
import torch.nn.functional as F
import numpy as np
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score
from tSNE import tsne_visualization
from utils import weight_to_vec

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for FedProx
FedProx_mu = 0.01

# for MOON
MOON_temperature = 0.5
MOON_mu = 1.0

class LSTM(torch.nn.Module):
    """
    Self-defined bi-LSTM model.
    """

    def __init__(self, args: object) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
        """

        super(LSTM, self).__init__()

        # LSTM
        input_size = 0
        input_size += 512 if 'emonet' in args.which_feature else 0
        input_size += 8 if 'openface_8' in args.which_feature else 0
        input_size += 14 if 'openface_14' in args.which_feature else 0
        input_size += 709 if 'openface' in args.which_feature else 0
        self.encoder = torch.nn.LSTM(input_size = input_size, hidden_size = args.h_size, num_layers = args.n_layer, bidirectional = args.bi_dir, batch_first = True)
        
        # logits layer
        logits_input_size = 2 * args.h_size if args.bi_dir else args.h_size
        logits_input_size += 512 if 'meglass' in args.which_feature else 0
        self.logits = torch.nn.Linear(in_features = logits_input_size, out_features = 2)

        # optimizer
        self.lr = args.client_lr
        self.optim = args.client_optim
        self.reuse_optim = args.reuse_optim
        self.optim_state = None
        
        # meglass feature
        self.use_meglass = 'meglass' in args.which_feature

        # imbalance weight
        self.imba_weight = args.imba_weight.to(device)
            
    def forward(self, x: torch.Tensor, meglass_mean: torch.Tensor = None, meglass_std: torch.Tensor = None) -> tuple[torch.Tensor]:
        """
        Arguments:
            x (torch.Tensor): input feature tensor.
            meglass_mean (torch.Tensor): mean of meglass feature.
            meglass_std (torch.Tensor): std of meglass feature.

        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
            h (torch.Tensor): latent features (useful for tSNE plot and some FL algorithms).
        """

        x, (hn, cn) = self.encoder(x)
        x = x[:, -1, :] # many-to-one LSTM
        
        # meglass feature
        if meglass_mean is not None and meglass_std is not None and self.use_meglass:
            x = torch.cat([x, meglass_mean, meglass_std], dim = 1)
        
        h = x            
        x = self.logits(x)
        x = torch.softmax(x, dim = 1)
        x = x[:, -1] # only need probability of class 1 (positive class)

        return x, h
    
class Dummy(torch.nn.Module):
    """
    Self-defined baseline classifier.
    """

    def __init__(self, args: object) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
        """

        super(Dummy, self).__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(6240, 3120),
            torch.nn.ReLU(),
            torch.nn.Linear(3120, 1560),
            torch.nn.ReLU(),
            torch.nn.Linear(1560, 780),
            torch.nn.ReLU(),
            torch.nn.Linear(780, 256),
            torch.nn.ReLU(),
        )
        
        self.logits = torch.nn.Sequential(
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.Softmax(dim = 1)
        )

        # optimizer
        self.lr = args.client_lr
        self.optim = args.client_optim
        self.reuse_optim = args.reuse_optim
        self.optim_state = None
        
        # meglass feature
        self.use_meglass = 'meglass' in args.which_feature

        # imbalance weight
        self.imba_weight = args.imba_weight.to(device)
            
    def forward(self, x: torch.Tensor, meglass_mean: torch.Tensor = None, meglass_std: torch.Tensor = None) -> tuple[torch.Tensor]:
        """
        Arguments:
            x (torch.Tensor): input feature tensor.
            meglass_mean (torch.Tensor): mean of meglass feature.
            meglass_std (torch.Tensor): std of meglass feature.

        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
            h (torch.Tensor): latent features (useful for tSNE plot and some FL algorithms).
        """

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # meglass feature
        if meglass_mean is not None and meglass_std is not None and self.use_meglass:
            x = torch.cat([x, meglass_mean, meglass_std], dim = 1)
        
        h = self.classifier(x)
        x = self.logits(h)
        x = x[:, -1] # only need probability of class 1 (positive class)
        return x, h

def model_train(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, num_client_epoch: int) -> None:
    """
    Train a model.

    Arguments:
        model (torch.nn.Module): pytorch model.
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        num_client_epoch (int): number of training epochs.
    """

    model.train()
    optimizer = model.optim(model.parameters(), lr = pow(10, model.lr))
    
    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optimizer.load_state_dict(model.optim_state)
    
    for current_client_epoch in range(num_client_epoch):
        for batch_id, (x, y, g, g_mean, g_std) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            g_mean = g_mean.to(device)
            g_std  = g_std .to(device)
            
            p, _ = model(x, g_mean, g_std)
            loss = F.binary_cross_entropy(p, y, weight = weight_to_vec(model.imba_weight, y))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # stability
            for p in model.parameters():
                torch.nan_to_num_(p.data, nan = 1e-6, posinf = 1e-6, neginf = 1e-6)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optimizer.state_dict())

# for FedProx
def model_train_FedProx(model: torch.nn.Module, global_model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, num_client_epoch: int) -> None:
    """
    Train a model when FedProx is chosen for federated learning.

    Arguments:
        model (torch.nn.Module): pytorch model (client model).
        global_model (torch.nn.Module): pytorch model (global model).
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        num_client_epoch (int): number of training epochs.
    """

    model.train()
    optim = model.optim(model.parameters(), lr = pow(10, model.lr))

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    for current_client_epoch in range(num_client_epoch):
        for batch_id, (x, y, g, g_mean, g_std) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            g_mean = g_mean.to(device)
            g_std  = g_std .to(device)
            
            p, _ = model(x, g_mean, g_std)
            loss = F.binary_cross_entropy(p, y, weight = weight_to_vec(model.imba_weight, y))
            
            # FedProx
            for p1, p2 in zip(model.parameters(), global_model.parameters()):
                ploss = (p1 - p2.detach()) ** 2
                loss += FedProx_mu * ploss.sum()

            loss.backward()
            optim.step()
            optim.zero_grad()

            # stability
            for p in model.parameters():
                torch.nan_to_num_(p.data, nan = 1e-6, posinf = 1e-6, neginf = 1e-6)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())

# for MOON
def model_train_MOON(model: torch.nn.Module, global_model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, previous_features: torch.Tensor) -> torch.Tensor:
    """
    Train a model when MOON is chosen for federated learning.

    Arguments:
        model (torch.nn.Module): pytorch model (client model).
        global_model (torch.nn.Module): pytorch model (global model).
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        previous_features (torch.Tensor): features extracted by client model in last global epoch.

    Returns:
        total_features (torch.Tensor): features extracted by client model in current global epoch.
    """

    model.train()
    optim = model.optim(model.parameters(), lr = pow(10, model.lr))

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    cos = torch.nn.CosineSimilarity(dim=-1)

    for batch_id, (x, y, g, g_mean, g_std) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        g_mean = g_mean.to(device)
        g_std  = g_std .to(device)
        
        p, features = model(x, g_mean, g_std)
        if batch_id == 0:
            total_features = torch.empty((0, features.size()[1]), dtype=torch.float32).to(device)
        total_features = torch.cat([total_features, features], dim=0)
        loss = F.binary_cross_entropy(p, y, weight = weight_to_vec(model.imba_weight, y))

        # for MOON
        features_tsne = np.squeeze(features)
        _, global_feat = global_model(x, g_mean, g_std)
        global_feat_copy = copy.copy(global_feat)
        posi = cos(features_tsne, global_feat_copy.to(device))
        logits = posi.reshape(-1,1)
        if previous_features == None or torch.count_nonzero(previous_features) == 0:
            previous_features = torch.zeros_like(features_tsne)
            nega = cos(features_tsne, previous_features)
            logits = torch.cat((posi.reshape(-1,1), nega.reshape(-1,1)), dim=1)
        if previous_features.dim() == 3:
            for prev_feat in previous_features[:, batch_id*y.size()[0]:(batch_id+1)*y.size()[0], :]:
                prev_nega = cos(features_tsne,prev_feat)
                logits = torch.cat((logits, prev_nega.reshape(-1,1)), dim=1)
        
        logits /= MOON_temperature # 0.5
        cos_labels = torch.zeros(logits.size(0)).long().to(device)
        loss_contrastive = F.cross_entropy(logits, cos_labels)
        if torch.count_nonzero(previous_features) != 0:
            loss += MOON_mu * loss_contrastive

        loss.backward()
        optim.step()
        optim.zero_grad()
        
        # stability
        for p in model.parameters():
            torch.nan_to_num_(p.data, nan = 1e-6, posinf = 1e-6, neginf = 1e-6)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())
    
    return total_features

def model_eval(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, wandb_log: dict, metric_prefix: str = 'prefix/', tSNE: bool = False) -> None:
    """
    Evaludate the performance of a model with differnt metrics (accuracy, MCC score, precision, recall, F1 score).

    Arguments:
        model (torch.nn.Module): pytorch model.
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        wandb_log (dict): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
        tSNE (bool): whether tSNE plot should be generated or not.
    """
    model.eval()
    epoch_labels = []
    epoch_preds  = []
    epoch_glass  = []
    epoch_feats  = []
    with torch.no_grad():
        for batch_id, (x, y, g, g_mean, g_std) in enumerate(data_loader):
                x = x.to(device)
                y = y.to(device)
                g = g.to(device)
                g_mean = g_mean.to(device)
                g_std  = g_std .to(device)
                
                p, f = model(x, g_mean, g_std)
                
                epoch_labels.append(y)
                epoch_preds .append(p)
                epoch_glass .append(g)

                f = np.squeeze(f.cpu().detach())
                for feat in f:
                    epoch_feats.append(feat)
            
    epoch_labels = torch.cat(epoch_labels).detach().to('cpu')
    epoch_preds  = torch.cat(epoch_preds ).detach().to('cpu')
    epoch_glass  = torch.cat(epoch_glass ).detach().to('cpu')
    
    # loss
    wandb_log[metric_prefix + 'loss'] = F.binary_cross_entropy(epoch_preds, epoch_labels, weight = weight_to_vec(model.imba_weight, epoch_labels))
    
    # ROC AUC not defined if there is only one class in truth labels
    try:
        wandb_log[metric_prefix + 'auc'] = roc_auc_score(epoch_labels, epoch_preds, average = 'weighted')
    except ValueError:
        wandb_log[metric_prefix + 'auc'] = 0
    
    # get class predictions
    epoch_preds = epoch_preds.round()

    # accuracy
    wandb_log[metric_prefix + 'accu'] = accuracy_score(epoch_labels, epoch_preds)

    # mcc
    wandb_log[metric_prefix + 'mcc'] = matthews_corrcoef(epoch_labels, epoch_preds)

    # precision
    wandb_log[metric_prefix + 'prec'] = precision_score(epoch_labels, epoch_preds, zero_division = 0, average = 'weighted')
    
    # recall
    wandb_log[metric_prefix + 'recl'] = recall_score(epoch_labels, epoch_preds, zero_division = 0, average = 'weighted')
    
    # f1 score
    wandb_log[metric_prefix + 'f1'] = f1_score(epoch_labels, epoch_preds, zero_division = 0, average = 'weighted')
    
    # f1 score for users with/without glasses
    if False: #'train' not in metric_prefix  and 'colorado' not in metric_prefix and 'engageNet' not in metric_prefix and 'daisee' not in metric_prefix and 'korea' not in metric_prefix:
        g  = epoch_glass == 1
        ng = epoch_glass == 0
        epoch_preds_g   = epoch_preds [g]
        epoch_labels_g  = epoch_labels[g]
        epoch_preds_ng  = epoch_preds [ng]
        epoch_labels_ng = epoch_labels[ng] 
        wandb_log[metric_prefix + 'g/f1' ] = f1_score(epoch_labels_g , epoch_preds_g , zero_division = 0, average = 'weighted')
        wandb_log[metric_prefix + 'ng/f1'] = f1_score(epoch_labels_ng, epoch_preds_ng, zero_division = 0, average = 'weighted')

    #t-SNE
    if tSNE:
        epoch_feats_array = np.zeros((len(epoch_feats), epoch_feats[0].size()[0]))
        for idx, element in enumerate(epoch_feats):
            epoch_feats_array[idx] = element
        try:
            tsne_visualization(epoch_feats_array, epoch_labels, metric_prefix, wandb_log, n_components = 2)
        except Exception:
            print('t-SNE failed!')