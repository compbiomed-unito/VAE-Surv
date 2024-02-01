import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sksurv
from sksurv.metrics import concordance_index_censored

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import optuna
import os
import sys
import torch
from torch import Tensor
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.VAE_surv import VAE
from model.loss import cox_ph_loss, cox_ph_loss_sorted




def rescale_dataset_genomed(cont_idx, x_train, x_test=None):
    
    cols_to_standard_scale = cont_idx #[58,61,62,63,64,65,66,69]

    scaler_standard = StandardScaler()
    x_train.iloc[:,cols_to_standard_scale] = scaler_standard.fit_transform(x_train.iloc[:,cols_to_standard_scale])
    if x_test is not None:
        x_test.iloc[:,cols_to_standard_scale]= scaler_standard.transform(x_test.iloc[:,cols_to_standard_scale])

    if x_test is not None:
        return x_train.values, x_test.values, scaler_standard
    else:
        return x_train.values, scaler_standard

    
    

def train_and_eval_model(train_loader, val_loader, params, dict_results=None, freeze_VAE=True, pretrained_vae=None):
    
    input_dim = train_loader.dataset.tensors[0].shape[1]
    output_dim = input_dim
    clin_dim = train_loader.dataset.tensors[1].shape[1]
    z_dim = params['z_dim']
    concat_dim = z_dim + clin_dim
    alpha = params['alpha']
    idx_cyto, idx_gen = 46,58 

    if pretrained_vae is None:
        
        pretrain=True
    else:
        pretrain=False
        

    model = VAE([input_dim, 64, 32], z_dim, [concat_dim,32,16,1],   
                        activation=params['activation'],
                        dropout_vae=params['dropout_vae'],
                        dropout_coxnet=params['dropout_coxnet'],
                        use_batchnorm=False,
                        KL_weight=params['KL_weight'],
                        loss_criterion='logcosh',
                        pretrain=pretrain)
    
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'],
                           weight_decay=0.)
    
    
    
    learn_curve_train = []
    learn_curve_test = []

    cox_curve_train = []
    vae_curve_train = []
    rec_curve_train = []
    kl_curve_train = []

    cox_curve_test = []
    vae_curve_test = []
    rec_curve_test = []
    kl_curve_test = []


    if pretrain:
        
        pretrain_epochs = params['pretrain_epochs']
        for epoch in range(pretrain_epochs):
            model.train()
            for gen,clin,event,time,idx in train_loader:
    
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                rec,mu,logvar,risk = model(gen, clin) 
                
                # Separate cytogenetics and gene mutations
                cyto = gen[:,idx_cyto:idx_gen]
                gen = gen[:,:idx_cyto]
                rec_gen = rec[:,:idx_cyto]
                rec_cyto = rec[:,idx_cyto:idx_gen]
                
                loss_vae, loss_rec, loss_kl = model.loss_function(rec_gen,rec_cyto,gen,cyto,mu,logvar)
                loss_cox = torch.tensor(0)
                loss = loss_vae
    
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            
            
            model.eval()
            for gen,clin,event,time,idx in val_loader:
    
                # Forward pass
                rec,mu,logvar,risk = model(gen,clin)
                
                # Separate cytogenetics and gene mutations
                cyto = gen[:,idx_cyto:idx_gen]
                gen = gen[:,:idx_cyto]
                rec_gen = rec[:,:idx_cyto]
                rec_cyto = rec[:,idx_cyto:idx_gen]
                
                loss_vae_test,loss_rec_test,loss_kl_test = model.loss_function(rec_gen,rec_cyto,gen,cyto,mu,logvar)
                
    
            
            if (epoch+1) % 10 == 0:
                print ('Pretrain: Epoch [{}/{}], Loss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(epoch+1,
                                                                                                      pretrain_epochs,
                                                                                                      loss.item(),
                                                                                                      loss_cox.item(),
                                                                                                      loss_vae.item(),
                                                                                                      loss_rec,
                                                                                                      loss_kl))
            
            learn_curve_train.append(loss.item())
            #learn_curve_test.append(loss_test.item())
    
            vae_curve_train.append(loss_vae.item())
            vae_curve_test.append(loss_vae_test.item())
            
            rec_curve_train.append(loss_rec.item())
            rec_curve_test.append(loss_rec_test.item())

            if params['KL_weight'] != 0:
                kl_curve_train.append(loss_kl.item())
                kl_curve_test.append(loss_kl_test.item())
    
            
        
        
    num_epochs = params['epochs']
    model.pretrain=False

    if pretrained_vae is not None:
        # Copy weights for encoding layers
        for current_layer, pretrained_layer in zip(model.encoding_layers, pretrained_vae.encoding_layers):
            current_layer.load_state_dict(pretrained_layer.state_dict())
    
        # Copy weights for fc_mu and fc_logvar
        model.fc_mu.load_state_dict(pretrained_vae.fc_mu.state_dict())
        model.fc_logvar.load_state_dict(pretrained_vae.fc_logvar.state_dict())
    
        # Copy weights for decoding layers
        for current_layer, pretrained_layer in zip(model.decoding_layers, pretrained_vae.decoding_layers):
            current_layer.load_state_dict(pretrained_layer.state_dict())


    coxnet_parameters = list(map(id, model.surv_layers.parameters()))
    vae_parameters = filter(lambda p: id(p) not in coxnet_parameters, model.parameters())
    coxnet_parameters = filter(lambda p: id(p) in coxnet_parameters, model.parameters())
    

    if not freeze_VAE:
        optimizer = optim.Adam([
        {"params": vae_parameters, "learning_rate":params['lr'], "weight_decay":params['weight_decay']},
        {"params": coxnet_parameters, "learning_rate":params['lr_coxnet'], "weight_decay": params['L2_coxnet']}])
    else:
        optimizer = optim.Adam(model.surv_layers.parameters(), lr=params['lr_coxnet'], weight_decay=params['L2_coxnet'])
        

    
    for epoch in range(params['epochs']):
        model.train()
        for gen,clin,event,time,idx in train_loader:

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            rec,mu,logvar,risk = model(gen,clin)
            
            # Separate cytogenetics and gene mutations
            cyto = gen[:,idx_cyto:idx_gen]
            gen = gen[:,:idx_cyto]
            rec_gen = rec[:,:idx_cyto]
            rec_cyto = rec[:,idx_cyto:idx_gen]
            
            loss_vae,loss_rec,loss_kl = model.loss_function(rec_gen,rec_cyto,gen,cyto,mu,logvar)
            loss_cox = cox_ph_loss(risk, time, event) 

            loss = (1-alpha)*loss_vae + alpha*loss_cox


            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        model.eval()
        for gen,clin,event,time,idx in val_loader:

            # Forward pass
            rec,mu,logvar,risk = model(gen,clin)
            
            # Separate cytogenetics and gene mutations
            cyto = gen[:,idx_cyto:idx_gen]
            gen = gen[:,:idx_cyto]
            rec_gen = rec[:,:idx_cyto]
            rec_cyto = rec[:,idx_cyto:idx_gen]
            
            loss_vae_test,loss_rec_test,loss_kl_test = model.loss_function(rec_gen,rec_cyto,gen,cyto,mu,logvar)
            loss_cox_test = cox_ph_loss(risk, time, event) 
            loss_test = (1-alpha)*loss_vae_test + alpha*loss_cox_test

           
        learn_curve_train.append(loss.item())
        learn_curve_test.append(loss_test.item())

        vae_curve_train.append(loss_vae.item())
        vae_curve_test.append(loss_vae_test.item())
        
        rec_curve_train.append(loss_rec.item())
        rec_curve_test.append(loss_rec_test.item())

        if params['KL_weight']!=0:
            kl_curve_train.append(loss_kl.item())
            kl_curve_test.append(loss_kl_test.item())

        cox_curve_train.append(loss_cox.item())
        cox_curve_test.append(loss_cox_test.item())



        # Print the loss every 10 epochs
        if (epoch+1) % 10 == 0:
            print ('Train: Epoch [{}/{}], Loss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(epoch+1, params['epochs'],
                                                                                               loss.item(), loss_cox.item(),
                                                                                               loss_vae.item(), loss_rec.item(), loss_kl.item()))
            print ('Test: Epoch [{}/{}], Loss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(epoch+1, params['epochs'],
                                                                                              loss_test.item(), loss_cox_test.item(),
                                                                                              loss_vae_test.item(), loss_rec_test.item(), loss_kl_test.item()), '\n')


            
    gen_train, gen_test = train_loader.dataset.tensors[0], val_loader.dataset.tensors[0]
    clin_train, clin_test = train_loader.dataset.tensors[1], val_loader.dataset.tensors[1]
    e_train, e_test = train_loader.dataset.tensors[2], val_loader.dataset.tensors[2]
    t_train, t_test = train_loader.dataset.tensors[3], val_loader.dataset.tensors[3]
    
    model.eval()
    risk_train = model(gen_train, clin_train)[-1].detach().numpy()
    risk_test = model(gen_test, clin_test)[-1].detach().numpy()
    
    CI_train = concordance_index_censored(np.bool_(e_train), t_train, risk_train.ravel())[0]
    CI_test = concordance_index_censored(np.bool_(e_test), t_test, risk_test.ravel())[0]
    
    print(CI_train)
    print(CI_test)
    
    if dict_results is not None:
    
        dict_results['CI_train'].append(CI_train)
        dict_results['learn_curve_train'].append(learn_curve_train)
        dict_results['cox_curve_train'].append(cox_curve_train)
        dict_results['vae_curve_train'].append(vae_curve_train)
        dict_results['rec_curve_train'].append(rec_curve_train)
        dict_results['kl_curve_train'].append(kl_curve_train)

        dict_results['CI_test'].append(CI_test)
        dict_results['learn_curve_test'].append(learn_curve_test)
        dict_results['cox_curve_test'].append(cox_curve_test)
        dict_results['vae_curve_test'].append(vae_curve_test)
        dict_results['rec_curve_test'].append(rec_curve_test)
        dict_results['kl_curve_test'].append(kl_curve_test)

        dict_results['best_params'].append(params)
        dict_results['model'].append(model)
    
        return model, dict_results
    
    else:
        return model, CI_test