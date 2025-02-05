import numpy
import pandas
from typing import List
#import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loss import cox_ph_loss, cox_ph_loss_sorted
from tqdm import tqdm
from sksurv.util import check_array_survival
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
import dataclasses

import sklearn

class NegativeBinomialLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        n = y_true
        p = torch.sigmoid(y_pred)
        return torch.mean(-n*torch.log(p + self.eps) - (1 - n)*torch.log(1 - p + self.eps))

class LogCoshLoss(torch.nn.Module):
    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


_losses = {
    'MSE': nn.MSELoss,
    'MAE': nn.L1Loss,
    'NBL': NegativeBinomialLoss,
    'LCH': LogCoshLoss,
}

def get_activation(activation_name: str):
    try:
        return getattr(nn, activation_name)()
    except AttributeError:
        raise ValueError(f"Invalid activation function '{activation_name}'")

def get_loss(loss_name: str):
    try:
        return _losses[loss_name]()
    except KeyError:
        raise ValueError(f"Invalid criterion '{loss_name}', must be one of {','.join(_losses)}")

def make_fnn_layers(dims, activation, dropout = 0.0, batchnorm = False):
    activation_mod = get_activation(activation)
    dropout_mod = nn.Dropout(dropout)
    layers = []
    for i, input_size in enumerate(dims[:-1]):
        is_hidden_layer = i < len(dims) - 2
        output_size = dims[i + 1]
        layers.append(nn.Linear(input_size, output_size))
        if is_hidden_layer:
            if batchnorm:
                layers.append(nn.BatchNorm1d(output_size))
            if dropout > 0.0:
                layers.append(dropout_mod)
            layers.append(activation_mod) # FIXME va prima activation o dropout?
    return nn.Sequential(*layers)


class VAE(nn.Module):
    def __init__(self, hidden_dims, latent_dim, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim

        encoder_dims = hidden_dims + [latent_dim * 2]
        self.encoder = make_fnn_layers(encoder_dims, **kwargs)
        decoder_dims = [latent_dim] + hidden_dims[::-1]
        self.decoder = make_fnn_layers(decoder_dims, **kwargs)

    def encode(self, x):
        z2 = self.encoder(x)
        mean = z2[..., :self.latent_dim]
        logvar = z2[..., self.latent_dim:]
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        softclamped_logvar = torch.clamp(logvar, min=-10.0, max=10.0)  #alternatively: torch.tanh(logvar / 10.0) * 10.0 
        stddev = torch.exp(0.5 * softclamped_logvar)
        z = mean + stddev*torch.randn_like(stddev) if self.training else mean # FIXME do we want this?
        r = self.decode(z)
        return dict(
            mean = mean,
            logvar = softclamped_logvar,
            stddev = stddev,
            zrep = z, # reparametrized z
            rec = r,
        )
    
class CoxNet(nn.Module):
    def __init__(self, hidden_dims, **kwargs):
        super().__init__()
        self.layers = make_fnn_layers(hidden_dims + [1], **kwargs)

    def forward(self, x):
        risk = self.layers(x) # FIXME originariamente ha tanh activation
        return risk

class TrainHistory:
    def __init__(self):
        self.history = []
        self.last_epoch = None
    
    def add_batch_losses(self, epoch, dataset, losses):
        # increasing epoch 
        if self.last_epoch is not None:
            assert epoch == self.last_epoch or epoch == self.last_epoch + 1, 'epoch must increase'
        self.last_epoch = epoch

        for loss_name, loss_value in losses.items():
            if hasattr(loss_value, 'item'):
                loss_value = loss_value.item()
            self.history.append([epoch, dataset, loss_name, loss_value])

    def get_df(self):
        return pandas.DataFrame.from_records(self.history, columns=['epoch', 'dataset', 'loss_name', 'loss_value'])
    

@dataclasses.dataclass
class VAESurv(sklearn.base.BaseEstimator):
    # VAESurv
    vae_hidden_dims: List[int] = dataclasses.field(default_factory=list)
    vae_latent_dim: int = 2
    vae_activation: str = 'Tanh'
    vae_dropout: float = 0.0
    vae_batchnorm: bool = False
    vae_loss: str = 'LCH'

    # survival
    survival_hidden_dims: List[int] = dataclasses.field(default_factory=list)
    survival_activation: str = 'Tanh'
    survival_dropout: float = 0.0
    survival_batchnorm: bool = False

    # other
    vae_feature_mask: numpy.ndarray = None
    alpha: float = 0.5
    kl_weight: float = 0.001

    # training
    vae_epochs: int = 10
    combo_epochs: int = 10
    survival_epochs: int = 0
    vae_learning_rate: float = 1e-4
    combo_learning_rate: float = 1e-4
    survival_learning_rate: float = 1e-4
    batch_size: int = 64

    vae_weight_decay: float = 0.
    survival_weight_decay: float = 0.
    combo_weight_decay: float = 0.

    # other
    device: str = 'cpu'
    verbose: int = 2

    def fit(self, X, y, X_test=None, y_test=None):
        # FIXME handle pandas dataframes

        # dataset prep
        #if self.vae_feature_mask is None:
            # use all features for the VAE
        #    self.vae_feature_mask = numpy.full(X.shape[1], True)
            
        if self.vae_feature_mask is None:
            self.vae_feature_mask = numpy.full(X.shape[1], True, dtype=bool)
        else:
            self.vae_feature_mask = numpy.array(self.vae_feature_mask, dtype=bool)
    
        vae_n_feats = self.vae_feature_mask.sum()
        surv_n_feats = (~self.vae_feature_mask).sum() + self.vae_latent_dim

        train_dataset = self._get_tensor_dataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        if X_test is not None:
            test_dataset = self._get_tensor_dataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            test_loader = None

        # model init
        self.modules_ = nn.ModuleDict({
            'VAE': VAE(
            [vae_n_feats] + self.vae_hidden_dims, self.vae_latent_dim, 
            activation=self.vae_activation,
            dropout=self.vae_dropout,
            batchnorm=self.vae_batchnorm,
        ),
            'survival': CoxNet(
            [surv_n_feats] + self.survival_hidden_dims,
            activation=self.survival_activation,
            dropout=self.survival_dropout,
            batchnorm=self.survival_batchnorm,
        ),
            'reconstruction_loss': get_loss(self.vae_loss),
        }).to(self.device)

        # model training
        self.train_history_ = TrainHistory()
        progress = tqdm if self.verbose >= 2 else lambda x, desc: x

        # VAE pretraining    
        optimizer = optim.Adam(
            self.modules_['VAE'].parameters(), 
            lr=self.vae_learning_rate,
            weight_decay=self.vae_weight_decay)
        for epoch in progress(range(self.vae_epochs), desc='VAE pretraining'):
            self._train(epoch, optimizer, train_loader, test_loader, train_survival=False)

        # combo training
        optimizer = optim.Adam(
            self.modules_.parameters(),
            lr=self.combo_learning_rate,
            weight_decay=self.combo_weight_decay)
        for epoch in progress(range(self.vae_epochs, self.vae_epochs + self.combo_epochs), desc='Training'):
            self._train(epoch, optimizer, train_loader, test_loader, train_survival=True)
            if test_loader is not None:
                risk_test = self.predict(X_test)
                self.train_history_.add_batch_losses(epoch, 'test', {
                    'c-index': concordance_index_censored(
                        event_indicator=y_test['event'],
                        event_time=y_test['time'],
                        estimate=risk_test,
                    )[0],
            })
        
        # survival training
        optimizer = optim.Adam(
            self.modules_['survival'].parameters(),
            lr=self.survival_learning_rate,
            weight_decay=self.survival_weight_decay)
        for epoch in progress(range(self.vae_epochs + self.combo_epochs, self.vae_epochs + self.combo_epochs + self.survival_epochs), desc='Post training'):
            self._train(epoch, optimizer, train_loader, test_loader, train_survival=True)
            # FIXME test_loader has shuffle=True, cannot use for c-index with the original y_test...
            if test_loader is not None:
                risk_test = self.predict(X_test)
                self.train_history_.add_batch_losses(epoch, 'test', {
                    'c-index': concordance_index_censored(
                        event_indicator=y_test['event'],
                        event_time=y_test['time'],
                        estimate=risk_test,
                    )[0],
            })


    def predict(self, X, return_all=False):
        results = self._run_model(X, run_survival=True)
        return results if return_all else results['risk'][:, 0]
        
    def transform(self, X, return_all=False):
        results = self._run_model(X, run_survival=False)
        return results if return_all else results['mean']

    def score(self, X, y):
        event, time = check_array_survival(X, y)
        return concordance_index_censored(event_indicator=event, event_time=time, estimate=self.predict(X))[0]

    def _get_tensor_dataset(self, X, y):
        # Ensure X is correctly formatted
        if isinstance(X, torch.Tensor):  
            X = X.clone().detach().cpu().numpy()  # Convert Tensor to NumPy to avoid slicing issues
        elif hasattr(X, 'values'):  # Pandas DataFrame
            X = X.to_numpy()  # Explicit conversion to NumPy
        

        Xvae = X[:, self.vae_feature_mask]
        Xsurvival = X[:, ~self.vae_feature_mask]

        ts = (
            torch.tensor(Xvae, dtype=torch.float32, device=self.device),
            torch.tensor(Xsurvival, dtype=torch.float32, device=self.device),
        )

        if y is not None:
            event, time = check_array_survival(X, y)
            ts += (
                torch.tensor(event.copy(), dtype=torch.bool, device=self.device), 
                torch.tensor(time.copy(), dtype=torch.float32, device=self.device),
            )

        return TensorDataset(*ts)


    def _run_model(self, X, run_survival):
        dataset = self._get_tensor_dataset(X, None)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.modules_.eval()
        result_acc = {}
        with torch.no_grad():
            for batch in loader:
                batch_results = self._run_model_batch(*batch, run_survival=run_survival)
                for name, values in batch_results.items():
                    result_acc.setdefault(name, []).append(values)
            results = {
                name: torch.concatenate(batches).cpu().numpy()
                for name, batches in result_acc.items()
            }
        return results
        
    def _run_model_batch(self, xvae, xother, run_survival):
        # FIXME option for not running decoder if we only want survival risk
        r = self.modules_['VAE'](xvae)
        if run_survival:
            z = r['zrep'] # reparametrized latent space?
            xsurv = torch.hstack([z, xother])
            r['risk'] = self.modules_['survival'](xsurv)
        return r

    def _compute_losses(self, batch, run_survival):
        xvae, xother, event, time = batch
        r = self._run_model_batch(xvae, xother, run_survival=run_survival)
        losses = {
            'reconstruction': self.modules_['reconstruction_loss'](xvae, r['rec']),
            'kl_divergence': -0.5 * torch.sum(
                1 + r['logvar'] - r['mean'].pow(2) - r['logvar'].exp()),
        }
        losses['vae'] = (1.0 - self.kl_weight) * losses['reconstruction'] + self.kl_weight * losses['kl_divergence']
        losses['survival'] = cox_ph_loss(r['risk'], time, event) if run_survival else torch.tensor(0.0)
        losses['total'] = (1.0 - self.alpha) * losses['vae'] + self.alpha * losses['survival']
        return losses, r

    def _train(self, epoch, optimizer, train_loader, test_loader, train_survival):
        self.modules_.train()
        for batch in train_loader:
            optimizer.zero_grad()
            losses, _ = self._compute_losses(batch, run_survival=train_survival)
            self.train_history_.add_batch_losses(epoch, 'train', losses)
            losses['total'].backward()
            optimizer.step()
        
        if test_loader is not None:
            self.modules_.eval()
            with torch.no_grad():
                for batch in test_loader:
                    losses, results = self._compute_losses(batch, run_survival=train_survival)
                    self.train_history_.add_batch_losses(epoch, 'test', losses)
 