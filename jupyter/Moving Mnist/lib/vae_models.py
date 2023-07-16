import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import math
import cv2
import torch.distributions as td

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims,kernel_size,stride):  
        super(VariationalEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size, stride, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride, padding=1)
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size, stride, padding=1)  
        self.conv4 = nn.Conv2d(64, 128, kernel_size, stride, padding=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size, stride, padding=1)
        self.linear1 = nn.Linear(256*2*2, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = td.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        
    def forward(self, x):
        x = x.to(device)


        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.batch3(self.conv4(x)))
        x = F.relu(self.conv5(x))
      
        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.linear1(x))

        mu =  self.linear2(x)
        log_var = self.linear3(x)
        sigma = torch.exp(0.5 * log_var)
        
        rand_state = torch.random.get_rng_state()[0]
        torch.random.manual_seed(rand_state+1)
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl=-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1)
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z   
    

class Decoder(nn.Module):
    
    def __init__(self, latent_dims,kernel_size,stride):
        super().__init__()
        self.latent_dims = latent_dims
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 256*2*2),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 2, 2))

        self.decoder_conv = nn.Sequential(

            nn.ConvTranspose2d(256, 128, kernel_size, stride, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size, stride, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size, stride, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size, stride, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size, stride, padding=1, output_padding=1),
            nn.Upsample(size=(64,64))
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        
        x = torch.sigmoid(x)
        return x
    
class GaussianDecoder(Decoder):
    def forward(self, x):
        x = self.decoder_lin(x)
        
        x = self.unflatten(x)
        x = self.decoder_conv(x)
    
        x = torch.sigmoid(x)
       
        var = torch.ones_like(x)
        
        return td.Normal(loc=x, scale=var)
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims,kernel_size,stride):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims,kernel_size,stride)
        self.decoder = GaussianDecoder(latent_dims,kernel_size,stride)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
### Training function
def vae_train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    nll_loss = 0.0
    kl_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x in dataloader: 
        # Move tensor to the proper device
        x = x[:,:1,:,:]
       
        x = x.to(device)

        px_z = vae(x)
        
        # Evaluate loss
        nll = -px_z.log_prob(x).sum((1,2,3)).mean()
     
        kl = vae.encoder.kl.mean()
  
        loss = nll + kl*0.1
        # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
      
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        nll_loss += nll.item()
        kl_loss += kl.item()

        train_loss+=loss.item()
        

    return train_loss / len(dataloader), nll_loss / len(dataloader), kl_loss / len(dataloader)

def vae_test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    nll_loss = 0.0
    kl_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x in dataloader:
            # Move tensor to the proper device
            x = x[:,:1,:,:]
            x = x.to(device)

           
            # Encode data
            # encoded_data = vae.encoder(x)
            # decoded_data = vae.decoder(encoded_data)
            # print(decoded_data.size())
            # Decode data
            px_z = vae(x)
        
        # Evaluate loss
            nll = -px_z.log_prob(x).sum((1,2,3)).mean()

            kl = vae.encoder.kl.mean()
            
            loss = nll + kl*0.1
            
            
            # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            nll_loss += nll.item()
            kl_loss += kl.item()

          
            val_loss += loss.item()
        
    return val_loss / len(dataloader), nll_loss / len(dataloader), kl_loss / len(dataloader)


def hi_vae_train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    nll_loss = 0.0
    kl_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for data in dataloader: 
        # Move tensor to the proper device
        x = data[:,:1,:,:]
        mask = data[:,1:,:,:]
        x = x.to(device)
        mask = mask.to(device)
        px_z = vae(x)

        temp_mask = mask.squeeze().flatten(1,2)
        mask_sum = temp_mask.sum(1)
        boolen_mask = (mask_sum == 0)
        temp_mask[boolen_mask] = 1
        temp_mask = temp_mask.reshape(mask.shape)
        
        # Evaluate loss
        nll = (-px_z.log_prob(x)*temp_mask).sum((1,2,3)).mean()

        kl = vae.encoder.kl.mean()

        loss = nll + kl*0.1
        # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
      
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        nll_loss += nll.item()
        kl_loss += kl.item()

        train_loss+=loss.item()
        

    return train_loss / len(dataloader), nll_loss / len(dataloader), kl_loss / len(dataloader)

def hi_vae_test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    nll_loss = 0.0
    kl_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for data in dataloader:
            # Move tensor to the proper device
            x = data[:,:1,:,:]
            mask = data[:,1:,:,:]
            x = x.to(device)
            mask = mask.to(device)

            temp_mask = mask.squeeze().flatten(1,2)
            mask_sum = temp_mask.sum(1)
            boolen_mask = (mask_sum == 0)
            temp_mask[boolen_mask] = 1
            temp_mask = temp_mask.reshape(mask.shape)

           
            # Encode data
            # encoded_data = vae.encoder(x)
            # decoded_data = vae.decoder(encoded_data)
            # print(decoded_data.size())
            # Decode data
            px_z = vae(x)
        
        # Evaluate loss
            nll = (-px_z.log_prob(x)*temp_mask).sum((1,2,3)).mean()

            kl = vae.encoder.kl.mean()

            loss = nll + kl*0.1
            
            
            # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            nll_loss += nll.item()
            kl_loss += kl.item()

          
            val_loss += loss.item()

    return val_loss / len(dataloader), nll_loss / len(dataloader), kl_loss / len(dataloader)

def ssim(img1, img2):
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2

    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def get_vae_summary_result(model,val_data,full_val_data):

    test_predict_loss = 0.0
    test_impu_part_loss = 0.0
    impuated_data_sum = 0
    test_psnr = 0
    test_ssim =0
    input_dataset = val_data
    compare_dataset = full_val_data
    with torch.no_grad():

        for i in range(len(input_dataset)):
            
            x = input_dataset[i][:1]
        
            y = compare_dataset[i].squeeze()

            input_image = x.unsqueeze(0)
    
            predict_image = model.decoder(model.encoder(input_image)).mean.squeeze().cpu()

            image_mask = input_dataset[i][1:].squeeze()
            impuated_data_sum+=(image_mask.flatten()==0).sum().item()
            imputed_part = predict_image*(1-image_mask)
            full_part = y*(1-image_mask)

            
            loss_p = (abs(predict_image - y)/(predict_image.size(-1)*predict_image.size(-2))).sum()
            loss_ip = (abs(imputed_part - full_part)).sum()
            mse= (((predict_image - y)**2)/(predict_image.size(-1)*predict_image.size(-2))).sum()
            
            
            psnr = 20*math.log10(1/math.sqrt(mse))
            test_psnr += psnr
        
            test_predict_loss += loss_p.item()
        
            test_impu_part_loss += loss_ip.item()
            ssim_p = ssim(predict_image.cpu().detach().numpy(),y.cpu().detach().numpy())
            test_ssim +=ssim_p
    return test_predict_loss / len(input_dataset), test_impu_part_loss/impuated_data_sum, test_psnr/len(input_dataset),test_ssim / len(input_dataset)

def get_bi_summary_result(model,trained_vae,val_data,full_val_data,mask_data,seqlen):
    test_predict_loss = 0.0
    test_impu_loss = 0.0
    test_impu_part_loss = 0.0
    impuated_data_sum = 0
    test_psnr = 0
    test_ssim =0
    input_dataset = val_data
    compare_dataset = full_val_data
    with torch.no_grad():
        for i in range(len(input_dataset)):

            x = input_dataset[i].to(device)
            y_img = compare_dataset[i].to(device)
            image_mask = mask_data[i].to(device)
            input_seq = torch.cat((x,image_mask),dim=1).unsqueeze(0)
        
            (pi, mu, sigma),predicts, imputation,latents = model(input_seq)
            
            predicts = trained_vae.decoder(latents.squeeze()).mean
            
            impuated_data_sum+=(image_mask.flatten()==0).sum().item()
            impuated_parts = predicts*(1-image_mask)
            full_parts = y_img*(1-image_mask)
            
            loss_p = (abs(predicts - y_img)/(predicts.size(-1)*predicts.size(-2))).sum()/seqlen
            loss_ip = abs(impuated_parts - full_parts).sum()
            mse = (((predicts- y_img)**2)/(predicts.size(-1)*predicts.size(-2))).sum()/seqlen

            psnr = 20*math.log10(1/math.sqrt(mse))

            test_psnr += psnr
        
            test_predict_loss += loss_p.item()
        
            test_impu_part_loss += loss_ip.item()
            ssim_p = 0
            for j in range(len(predicts.squeeze())):
                ssim_p += ssim(predicts.squeeze()[j].squeeze().cpu().detach().numpy(),y_img.squeeze()[j].squeeze().cpu().detach().numpy())
            test_ssim +=ssim_p/seqlen
    return test_predict_loss / len(input_dataset),test_impu_part_loss / impuated_data_sum, test_psnr / len(input_dataset), test_ssim / len(input_dataset)



class MDNRNN(nn.Module):
    def __init__(self, z_size,vae_trained, n_hidden=256, n_gaussians=10, n_layers=1,direction='forward'):
        super(MDNRNN, self).__init__()

        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        self.direction = direction
        self.vae_trained = vae_trained
        self.lstm_cell = nn.LSTMCell(z_size,n_hidden)
        self.lstm = nn.LSTM(z_size, n_hidden, n_layers, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc2 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc3 = nn.Linear(n_hidden, n_gaussians*z_size)
        
    def get_mixture_coef(self, y):
     
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, self.n_gaussians, self.z_size)
        mu = mu.view(-1, self.n_gaussians, self.z_size)
        sigma = sigma.view(-1, self.n_gaussians, self.z_size)

        pi = F.softmax(pi, 1)
        sigma = torch.exp(sigma)
        sigma = abs(sigma)
        return pi, mu, sigma
        
        
    def forward(self, data ,direct=None):

        if not direct:
            direct = self.direction
        
        if direct == "backward":
            data = data.flip(1)
       
        
        input_d = data[:,:,0,:,:] #size(10,6,64,64)
        input_m = data[:,:,1,:,:]

        output_pi = []
        output_mu = []
        output_sigma = []

        predict_result = []
        imputed_result = []
        latent_result = []

        h_c = (torch.zeros(data.size(0), self.n_hidden).to(device),
                torch.zeros(data.size(0), self.n_hidden).to(device))

        for t in range(input_d.size(1)):
            # get predict img by hidden
            h,_ = h_c
            pi, mu, sigma = self.get_mixture_coef(h)

            output_pi.append(pi)
            output_mu.append(mu)
            output_sigma.append(sigma)
          
            pi = pi.unsqueeze(1)
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
            
        
            # preds=torch.sum((torch.normal(mu, sigma)*pi),dim=2).squeeze(1)
            # preds=torch.normal(mu, sigma).gather(2,pi.max(2)[1].unsqueeze(2)).squeeze(2).squeeze(1)
           
            pi_t=pi.transpose(2, 3)
            pi_t_r=pi_t.reshape(pi_t.size(0)*pi_t.size(1)*pi_t.size(2),pi_t.size(3))
            
            rand_state = torch.random.get_rng_state()[0]
            torch.random.manual_seed(rand_state)

            k = torch.multinomial(pi_t_r, 1).view(-1)
            
            k_r = k.reshape(pi_t.size(0),pi_t.size(1),pi_t.size(2)).unsqueeze(2)

            torch.random.manual_seed(rand_state+1)
            preds = torch.normal(mu, sigma).gather(2,k_r).squeeze(2).squeeze(1)
            with torch.no_grad():
                preds_img = self.vae_trained.decoder(preds).mean
                
            
            predict_result.append(preds_img)
            # combine input image and predict image
            input_t = input_d[:,t,:,:].unsqueeze(1)
            m = input_m[:,t,:,:].unsqueeze(1)
            input_c = input_t*m+preds_img*(1-m)
            imputed_result.append(input_c)
            # Get latent dimension by vae
            with torch.no_grad():
                x = self.vae_trained.encoder(input_c)

            latent_result.append(x)
            h_c = self.lstm_cell(x,h_c)
            
           

        # predict next image use for imputation next target

      
        
        pi_t = torch.stack(output_pi, dim=1)
        mu_t = torch.stack(output_mu, dim=1)
        sigma_t = torch.stack(output_sigma, dim=1)
        imputations = torch.stack(imputed_result, dim=1)
        predicts= torch.stack(predict_result, dim=1)
        latents = torch.stack(latent_result, dim=1).unsqueeze(2)
        if direct == "backward":
            pi_t=pi_t.flip(1)
            mu_t=mu_t.flip(1)
            sigma_t=sigma_t.flip(1)
            imputations=imputations.flip(1)
            predicts=predicts.flip(1)
            latents = latents.flip(1)
        return (pi_t,mu_t,sigma_t),predicts,imputations,latents

class BIMDNRNN(nn.Module):
    def __init__(self, z_size,vae_trained, n_hidden=256, n_gaussians=10, n_layers=1 ):
        super(BIMDNRNN, self).__init__()

        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers

        

        self.vae_trained = vae_trained
        self.mdnrnn_f = MDNRNN(z_size,vae_trained, n_hidden,n_gaussians)
        self.mdnrnn_b = MDNRNN(z_size,vae_trained, n_hidden,n_gaussians)
    
    def forward(self,data):
        mdnrnn_f = self.mdnrnn_f(data,'forward')
        mdnrnn_b = self.mdnrnn_b(data,'backward')
        mask =  data[:,:,1,:,:]
        maskweight = torch.sum(mask,(2,3))
        seqlen = data.size(1)
        d_weight = maskweight.unsqueeze(2)
        
        d_weight_reverse = d_weight.flip(1)
        df_cum_weight = np.cumsum(d_weight.cpu(),1).to(device)
        db_cum_weight = np.cumsum(d_weight_reverse.cpu(),1).flip(1).to(device)
        
        return self.merge_fnb(mdnrnn_f,mdnrnn_b,df_cum_weight,db_cum_weight,seqlen)


    def merge_fnb(self,mdnrnn_f,mdnrnn_b,df_cum_weight,db_cum_weight,seqlen):
        
        (pi_f, mu_f, sigma_f), predicts_f,imputation_f,latents_f = mdnrnn_f

        (pi_b, mu_b, sigma_b), predicts_b,imputation_b,latents_b = mdnrnn_b
        

        size_view = pi_f.size()
        
       
        
        pi_m = torch.cat(((pi_f.flatten(2)*df_cum_weight/(df_cum_weight+db_cum_weight)).view(size_view),(pi_b.flatten(2)*db_cum_weight/(df_cum_weight+db_cum_weight)).view(size_view)),dim=2)
        
        mu_m = torch.cat((mu_f,mu_b),dim=2)
        
        sigma_m = torch.cat((sigma_f,sigma_b),dim=2)
        predict_size = predicts_f.size()
        imputaion_size = imputation_f.size()
        latent_size = latents_f.size()

    
        
        predicts_m = ((predicts_f.flatten(2)*df_cum_weight + predicts_b.flatten(2)*db_cum_weight)/(df_cum_weight+db_cum_weight)).view(predict_size)
        imputation_m = ((imputation_f.flatten(2)*df_cum_weight + imputation_b.flatten(2)*db_cum_weight)/(df_cum_weight+db_cum_weight)).view(imputaion_size)
        latents_m = ((latents_f.flatten(2)*df_cum_weight + latents_b.flatten(2)*db_cum_weight)/(df_cum_weight+db_cum_weight)).view(latent_size)
        # latents_m = []
        # with torch.no_grad():
        #     for i in range(seqlen):
        #         latents_m.append(self.vae_trained.encoder(imputation_m[:,i,:]))
        #     latents_m = torch.stack(latents_m, dim=1).unsqueeze(2)

        return (pi_m,mu_m,sigma_m),predicts_m,imputation_m,latents_m
    
def mdn_loss_fn(y, pi, mu, sigma):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim=2)
    loss = -torch.log(loss)
    return loss.mean()

class GPVariationalEncoder(nn.Module):
    def __init__(self, latent_dims,kernel_size,stride):  
        super(GPVariationalEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.conv1 = nn.Conv2d(1, 16, kernel_size, stride, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride, padding=1)
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size, stride, padding=1)  
        self.conv4 = nn.Conv2d(64, 128, kernel_size, stride, padding=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size, stride, padding=1)
        self.linear1 = nn.Linear(256*2*2, 128)

        self.conv1d = nn.Conv1d(128,128, kernel_size=3,padding="same")

        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        

        self.N = td.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        
    def forward(self, x):
        seqlen = x.size(1)
        x = x.to(device)
        x = x.view(-1,1,64,64)

        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.batch3(self.conv4(x)))
        x = F.relu(self.conv5(x))
        
        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.linear1(x))
      
        x = x.view(-1,seqlen,128)
        x = x.permute(0,2,1)
        
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        
        mu =  self.linear2(x)
        mu = mu.permute(0,2,1)
        sigma = torch.exp(self.linear3(x))
        sigma = sigma.permute(0,2,1)
        return td.Independent(td.Normal(
                    mu,
                    sigma),1)
    
class GPGaussianDecoder(Decoder):
    def forward(self, x):
        seqlen = x.size(-1)
        x = x.permute(0,2,1).view(-1,self.latent_dims)
        x = self.decoder_lin(x)
        
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = x.view(-1,seqlen,1,64,64)
        x = torch.sigmoid(x)
       
        var = torch.ones_like(x)
        
        return td.Normal(loc=x, scale=var)
    
def rbf_kernel(T, length_scale):
    xs = torch.arange(T, dtype=torch.float32)
    xs_in = xs.unsqueeze(0)
    xs_out = xs.unsqueeze(1)
    distance_matrix = torch.square(torch.sub(xs_in, xs_out))
    distance_matrix_scaled = distance_matrix / torch.square(length_scale)
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def diffusion_kernel(T, length_scale):
    assert length_scale < 0.5, "length_scale has to be smaller than 0.5 for the "\
                               "kernel matrix to be diagonally dominant"
    sigmas = torch.ones(T, T, dtype=torch.float32) * length_scale
    sigmas_tridiag = sigmas.diag(-1) + sigmas.diag() + sigmas.diag(1)
    kernel_matrix = sigmas_tridiag + torch.eye(T)*(1. - length_scale)
    return kernel_matrix


def matern_kernel(T, length_scale):
    xs = torch.arange(T, dtype=torch.float32)
    xs_in = xs.unsqueeze(0)
    xs_out = xs.unsqueeze(1)
    distance_matrix = torch.abs(torch.sub(xs_in, xs_out))
    distance_matrix_scaled = distance_matrix / torch.sqrt(length_scale)
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def cauchy_kernel(T, sigma, length_scale):
    xs = torch.arange(T, dtype=torch.float32)
    xs_in = xs.unsqueeze(0)
    xs_out = xs.unsqueeze(1)
    distance_matrix = torch.square(torch.sub(xs_in, xs_out))
 
    distance_matrix_scaled = distance_matrix / torch.square(torch.tensor(length_scale))
    kernel_matrix = torch.div(sigma, (distance_matrix_scaled + 1.))

    alpha = 0.001
    eye = torch.eye(kernel_matrix.shape[-1])
    return kernel_matrix + alpha * eye

class GPVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims,kernel_size,stride,kernel="cauchy",time_length = 16, sigma=1., length_scale=1.0, kernel_scales=1):
        super(GPVariationalAutoencoder, self).__init__()
        self.time_length = time_length
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales
        self.latent_dims = latent_dims
   

        # Precomputed KL components for efficiency
        self.pz_scale_inv = None
        self.pz_scale_log_abs_determinant = None
        self.prior = None
        self.encoder = GPVariationalEncoder(latent_dims,kernel_size,stride)
        self.decoder = GPGaussianDecoder(latent_dims,kernel_size,stride)
    def forward(self, x):
        x = x.to(device)
        
        pz = self._get_prior()
        
        qz_x = self.encoder(x)
        
        kl = self.kl_divergence(qz_x,pz)
      
        z = qz_x.rsample()
        
        return self.decoder(z), kl
    
    def _get_prior(self):
        if self.prior is None:
            # Compute kernel matrices for each latent dimension
            kernel_matrices = []
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(rbf_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "diffusion":
                    kernel_matrices.append(diffusion_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(matern_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2**i))
          
            # Combine kernel matrices for each latent dimension
            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.latent_dims - total
                else:
                    multiplier = int(np.ceil(self.latent_dims / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(torch.tile(kernel_matrices[i].unsqueeze(0), [multiplier, 1, 1]))
            kernel_matrix_tiled = torch.cat(tiled_matrices)
            assert len(kernel_matrix_tiled) == self.latent_dims

            self.prior = td.MultivariateNormal(
                loc=torch.zeros(self.latent_dims, self.time_length),
                covariance_matrix=kernel_matrix_tiled)
           
            
        return self.prior
    def kl_divergence(self, a, b):
        
        def squared_frobenius_norm(x):
            """Helper to make KL calculation slightly more readable."""
            return torch.sum(torch.square(x), dim=[-2, -1])

      
        if self.pz_scale_inv is None:
            
            self.pz_scale_inv = torch.linalg.inv(b.scale_tril.to_sparse().to_dense())
            
            self.pz_scale_inv = torch.where(torch.isfinite(self.pz_scale_inv),
                                            self.pz_scale_inv, torch.zeros_like(self.pz_scale_inv))
       
        if self.pz_scale_log_abs_determinant is None:
           
            

            self.pz_scale_log_abs_determinant = torch.logdet(b.scale_tril)
       
        a_scale =  torch.linalg.cholesky(torch.diag_embed(a.stddev))
        a_shape = a_scale.shape
        
        if len(b.scale_tril.shape) == 3:
            _b_scale_inv = self.pz_scale_inv[None].repeat(a_shape[0], *[1] * (len(a_shape) - 1))
        else:
            _b_scale_inv = self.pz_scale_inv.repeat(a_shape[0], *[1] * (len(a_shape) - 1))
        _b_scale_inv = _b_scale_inv.to(device)

        b_inv_a = _b_scale_inv @ a_scale.to_sparse().to_dense().to(device)
        kl_div = (self.pz_scale_log_abs_determinant.to(device) - torch.logdet(a_scale).to(device) +
                0.5 * (-torch.tensor(self.time_length).to(device) +
                squared_frobenius_norm(b_inv_a).to(device) + squared_frobenius_norm(
                torch.linalg.solve(b.scale_tril.to(device),(b.mean.to(device) - a.mean.to(device))[..., None])).to(device)))
      
        return kl_div
    
def gp_train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    nll_loss = 0.0
    kl_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, mask in dataloader: 
        # Move tensor to the proper device
    
        x = x.to(device)
        mask = mask.to(device)

        px_z,kl = vae(x)

        temp_mask = mask.squeeze().flatten(0,1).flatten(1,2)
        mask_sum = temp_mask.sum(1)
        boolen_mask = (mask_sum == 0)
        temp_mask[boolen_mask] = 1
        temp_mask = temp_mask.reshape(mask.shape)
    
        # Evaluate loss
        nll = (-px_z.log_prob(x)*temp_mask).sum((2,3,4)).mean()
        
        kl = kl.sum((1)).mean()

        loss = nll + kl*0.1
        
        # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
    
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        nll_loss += nll.item()
        kl_loss += kl.item()

        train_loss+=loss.item()
        

    return train_loss / len(dataloader), nll_loss / len(dataloader), kl_loss / len(dataloader)

def gp_test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    nll_loss = 0.0
    kl_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, mask in dataloader:
         
            # Move tensor to the proper device
          
            x = x.to(device)
            mask = mask.to(device)
            px_z,kl = vae(x)

            temp_mask = mask.squeeze().flatten(1,2)
            mask_sum = temp_mask.sum(1)
            boolen_mask = (mask_sum == 0)
            temp_mask[boolen_mask] = 1
            temp_mask = temp_mask.reshape(mask.shape)
            
            # Evaluate loss
            nll = (-px_z.log_prob(x)*temp_mask).sum((2,3,4)).mean()
        
            kl = kl.sum((1)).mean()

            loss = nll + kl*0.1

            # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            nll_loss += nll.item()
            kl_loss += kl.item()

            val_loss += loss.item()

    return val_loss / len(dataloader), nll_loss / len(dataloader), kl_loss / len(dataloader)

def gp_vae_summary_result(vae,x_val_miss,x_val_full,m_val_miss,seqlen):
    test_predict_loss = 0.0
    test_impu_loss = 0.0
    test_impu_part_loss = 0.0
    impuated_data_sum = 0
    test_psnr = 0
    test_ssim =0
    input_dataset = x_val_miss
    compare_dataset = x_val_full
    with torch.no_grad():

        for i in range(len(x_val_miss)):
            
            x = input_dataset[i]
        
            y = compare_dataset[i]
        
            input_image = x.unsqueeze(0)
            
            predict_image = vae.decoder(vae.encoder(input_image).sample()).mean.cpu()
            
            image_mask = m_val_miss[i]
            impuated_data_sum+=(image_mask.flatten()==0).sum().item()
            imputed_part = predict_image*(1-image_mask)
        
            full_part = y*(1-image_mask)
        
        
            loss_p = (abs(predict_image - y)/(predict_image.size(-1)*predict_image.size(-2))).sum()/seqlen
            loss_ip = (abs(imputed_part - full_part)).sum()
            mse= (((predict_image - y)**2)/(predict_image.size(-1)*predict_image.size(-2))).sum()/seqlen
            
            
            psnr = 20*math.log10(1/math.sqrt(mse))
            test_psnr += psnr
        
            test_predict_loss += loss_p.item()
        
            test_impu_part_loss += loss_ip.item()
            ssim_p = 0
        
            for j in range(len(predict_image.squeeze())):
                ssim_p += ssim(predict_image.squeeze()[j].squeeze().cpu().detach().numpy(),y.squeeze()[j].squeeze().cpu().detach().numpy())
            test_ssim += ssim_p/seqlen
    return  test_predict_loss / len(input_dataset), test_impu_part_loss / impuated_data_sum,test_psnr / len(input_dataset),test_ssim / len(input_dataset)


