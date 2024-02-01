import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.optim as optim


class NegativeBinomialLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        n = y_true
        p = torch.sigmoid(y_pred)
        return torch.mean(-n*torch.log(p + self.eps) - (1 - n)*torch.log(1 - p + self.eps))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

    
    
    
class VAE(nn.Module):
    def __init__(self, encoding_nodes, latent_dim, surv_nodes, activation, dropout_vae, dropout_coxnet, use_batchnorm,
                 pretrain, KL_weight, loss_criterion):
        super(VAE, self).__init__()
        
        self.encoder_dims = encoding_nodes
        self.surv_dims = surv_nodes
        self.pretrain = pretrain
        self.K = KL_weight
        self.use_batchnorm = use_batchnorm
        self.latent_dim = latent_dim
        
        self.num_genetic = 58 

        
        if self.pretrain == True:
            print('Pretraining Autoencoder: True')
        
        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()
        self.surv_layers = nn.ModuleList()
        
        
        # Add activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'leaky relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError("Invalid activation function")
            
        
        if loss_criterion == 'mse':
            self.criterion = nn.MSELoss()
        if loss_criterion == 'mae':
            self.criterion = nn.L1Loss()
        if loss_criterion == 'nbl':
            self.criterion = NegativeBinomialLoss()
        if loss_criterion == 'logcosh':
            self.criterion = LogCoshLoss()
        
        # Add dropout
        self.dropout_vae = nn.Dropout(dropout_vae)
        self.dropout_coxnet = nn.Dropout(dropout_coxnet)
        
        # Add encoding layers
        num_layers = len(self.encoder_dims)
        
        for i in range(num_layers-1):
            self.encoding_layers.append(nn.Linear(self.encoder_dims[i], self.encoder_dims[i+1]))
            
            #if self.use_batchnorm:
            #    self.encoding_layers.append(nn.BatchNorm1d(self.encoder_dims[i+1]))
            self.encoding_layers.append(self.activation)
            self.encoding_layers.append(self.dropout_vae)
            
        self.fc_mu = nn.Linear(self.encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_dims[-1], latent_dim)
        
        
        
        
        # Add decoding layers
        self.decoding_layers.append(nn.Linear(latent_dim, self.encoder_dims[-1]))
        self.decoding_layers.append(self.dropout_vae)
        
        for i in range(num_layers-1, 0, -1):
            self.decoding_layers.append(nn.Linear(self.encoder_dims[i], self.encoder_dims[i-1]))
            #if self.use_batchnorm:
            #        self.decoding_layers.append(nn.BatchNorm1d(self.encoder_dims[i-1]))
            self.decoding_layers.append(self.activation)
            
            
            if i != 1:
                self.decoding_layers.append(self.dropout_vae)
            
            
            
        if self.surv_dims is not None:
            # Add coxnet layers
            num_surv_layers = len(self.surv_dims) - 1
            for i in range(num_surv_layers):
                self.surv_layers.append(nn.Linear(self.surv_dims[i], self.surv_dims[i+1]))
                #if i==0:
                #    self.surv_layers.append(nn.BatchNorm1d(self.surv_dims[i+1]))


    
    def encode(self, x):
        for layer in self.encoding_layers:
            x = layer(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    # In the decode function
    def decode(self, z):
        for layer in self.decoding_layers:
            z = layer(z)
        return z
    

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
    
    
    
    def coxnet(self, x_concat):
        for i, layer in enumerate(self.surv_layers):
            x_concat = layer(x_concat)
            if i != len(self.surv_layers) - 1:
                x_concat = self.dropout_coxnet(x_concat) #self.dropout

            # apply tanh activation if this is the last layer, else use the predefined activation
            if i == len(self.surv_layers) - 1:
                x_concat = nn.Tanh()(x_concat)
            else:
                x_concat = nn.Tanh()(x_concat) #nn.Tanh()(x_concat) #self.activation(x_concat)

        x_concat = torch.exp(x_concat)

        return x_concat

    
    
    def forward(self, x, c):
        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, max=10)
            
        z = self.reparameterize(mu, logvar)
        
        # Standardize z
        #mu_z = torch.mean(z, dim=0, keepdim=True)
        #std_z = torch.std(z, dim=0, keepdim=True)
        #z_standardized = (z - mu_z) / (std_z + 1e-9)
        
        if self.pretrain is False and c is not None:
            x_concat = torch.hstack([z,c]) 
            risk = self.coxnet(x_concat)
            
        else:
            risk=None
            
        rec = self.decode(z)
        
        return rec, z, logvar, risk 
    
    
    def predict_risk(self, data):
        gen = torch.tensor(data[:,:self.num_genetic], dtype = torch.float32)
        clin = torch.tensor(data[:,self.num_genetic:], dtype = torch.float32)
        return self.forward(gen, clin)[-1]  ###### 1 ######
    
    
    
    def loss_function(self, rec_gen, rec_cyto, x_gen, x_cyto, mu, logvar):
        # Compute reconstruction loss
        reconstruction_loss = self.criterion(rec_gen,x_gen) + self.criterion(rec_cyto,x_cyto)
        
        # Compute KL divergence loss
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence = self.K*kl_divergence
            
       
        # Combine losses
        loss = reconstruction_loss + kl_divergence
        
        return loss, reconstruction_loss, kl_divergence  
 