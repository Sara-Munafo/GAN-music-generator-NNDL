import mido
from mido import MidiFile, MidiTrack, Message
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os
import random


from scipy.special import kl_div
from scipy.stats import entropy


####################################################################################################################################################
                                                              # GAN model architecture
###################################################################################################################################################

def conv_prev_concat(x, y):
    """
    Concatenate conditioning vector on feature map axis.
    Parameters:
      x (torch.Tensor) : first vector
      y (torch.Tensor) : second vector
    Returns:
      concatenated vector
    """
    x_shapes = x.shape
    y_shapes = y.shape
    if x_shapes[2:] == y_shapes[2:]:
        y2 = y.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])

        return torch.cat((x, y2),1)

    else:
        print(x_shapes[2:])
        print(y_shapes[2:])


def batch_norm_1d_sampl(x):
    output = x
    return output


def one_hot(x):
    '''
    Applied after activation of the generator to produce one-hot encoded output
    Parameters:
      x (torch.tensor) : vector after activation of last layer
    Returns:
      one_hot_x (torch.tensor) : one hot encoded output
    '''
    max_indices = torch.argmax(x, dim=-1, keepdim=True)
    one_hot_x = torch.zeros_like(x)
    one_hot_x.scatter_(-1, max_indices, 1)
    return one_hot_x
    
    
    
    
# Self attention module  
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim # 1
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1) #(batch_size,1,16,128)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)   #(batch_size,1,16,128)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H) (batch_size, 1, 16, 128)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batch_size, C , width ,height = x.size()
        proj_query  = self.query_conv(x).view(batch_size,-1,width*height).permute(0,2,1) # B X CX(N)    # (batchs​ize,1,2048) --> (batchs​ize,2048,1)
        proj_key =  self.key_conv(x).view(batch_size,-1,width*height) # B X C x (*W*H)                  # (batchs​ize,1,2048)
        score =  torch.bmm(proj_query,proj_key) # transpose check  					# (batchs​ize,2048,2048)
        attention = self.softmax(score) # BX (N) X (N)   						# (batchs​ize,2048,2048)
        proj_value = self.value_conv(x).view(batch_size,-1,width*height) # B X C X N                    # (batchs​ize,1,2048)

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(batch_size,C,width,height)                                                       # (batchs​ize,1,16,128)

        out = self.gamma*out + x                                                                        # (batchs​ize,1,16,128)
        return out,attention



# 1. GENERATOR
class generator(nn.Module):
    def __init__(self, pitch_range):
        super(generator, self).__init__()
        self.gf_dim = 64
        self.n_channel = 256

        self.h1 = nn.ConvTranspose2d(in_channels=144, out_channels=pitch_range, kernel_size=(2, 1), stride=(2, 2))
        self.h2 = nn.ConvTranspose2d(in_channels=144, out_channels=pitch_range, kernel_size=(2, 1), stride=(2, 2))
        self.h3 = nn.ConvTranspose2d(in_channels=144, out_channels=pitch_range, kernel_size=(2, 1), stride=(2, 2))
        self.h4 = nn.ConvTranspose2d(in_channels=144, out_channels=1, kernel_size=(1, pitch_range), stride=(1, 2))

        self.h0_prev = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, pitch_range), stride=(1, 2))
        self.h1_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2))
        self.h2_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2))
        self.h3_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2))

        self.linear1 = nn.Linear(100, 1024)
        self.linear2 = nn.Linear(1024, self.gf_dim * 2 * 2)

        self.bn1d_1 = nn.BatchNorm1d(1024, eps=1e-05, momentum=0.9, affine=True)
        self.bn1d_2 = nn.BatchNorm1d(self.gf_dim *2*2, eps=1e-05, momentum=0.9, affine=True)

        self.bn2d_0 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True)
        self.bn2d_1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True)
        self.bn2d_2 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True)
        self.bn2d_3 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True)
        self.bn2d_4 = nn.BatchNorm2d(pitch_range, eps=1e-05, momentum=0.9, affine=True)

        self.attn1 = Self_Attn( 144, 'relu')
        self.attn2 = Self_Attn( 144,  'relu')

    def forward(self, z, prev_x, batch_size, pitch_range, attention):
        h0_prev = F.leaky_relu(self.bn2d_0(self.h0_prev(prev_x)), 0.2)  #[batch_size, 16, 16, 1]
        h1_prev = F.leaky_relu(self.bn2d_1(self.h1_prev(h0_prev)), 0.2) #[batch_size, 16, 8, 1]
        h2_prev = F.leaky_relu(self.bn2d_2(self.h2_prev(h1_prev)), 0.2) #[batch_size, 16, 4, 1]
        h3_prev = F.leaky_relu(self.bn2d_3(self.h3_prev(h2_prev)), 0.2) #[batch_size, 16, 2, 1]

        h0 = F.relu(self.bn1d_1(self.linear1(z)))                       #(batch_size,1024)
        h1 = F.relu(self.bn1d_2(self.linear2(h0)))                      #(batch_size,1024)
        h1 = h1.view(batch_size, self.gf_dim * 2, 2, 1)                 #(batch_size,128,2,1)

        h1 = conv_prev_concat(h1, h3_prev)                              #(batch_size,144,2,1)
        h2 = F.relu(self.bn2d_4(self.h1(h1))) 				#([batch_size, 128, 4, 1])
        h2 = conv_prev_concat(h2, h2_prev)    				#([batch_size, 144, 4, 1])

        if attention:
          h2,p1 = self.attn1(h2)                			#([batch_size, 144, 4, 1])  Attention

        h3 = F.relu(self.bn2d_4(self.h2(h2))) 				#([batch_size, 128, 8, 1])
        h3 = conv_prev_concat(h3, h1_prev)    				#([batch_size, 144, 8, 1])
        h4 = F.relu(self.bn2d_4(self.h3(h3))) 				#([batch_size, 128, 16, 1])
        h4 = conv_prev_concat(h4, h0_prev)    				#([batch_size, 144, 16, 1])

        if attention:
          h4,p2 = self.attn1(h4)                			#([batch_size, 144, 16, 1]) Attention


        g_x = torch.sigmoid(self.h4(h4))      				#([batch_size, 1, 16, 128])
        g_x = one_hot(g_x)

        return g_x


# 2. DISCRIMINATOR
class discriminator(nn.Module):
    def __init__(self, pitch_range):
        super(discriminator, self).__init__()

        self.df_dim = 64
        self.dfc_dim = 1024

        # Define the convolutional and linear layers
        self.h0_prev = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, pitch_range), stride=(2, 2))
        self.h1_prev = nn.Conv2d(in_channels=1, out_channels=77, kernel_size=(4, 1), stride=(2, 2))

        self.linear1 = nn.Linear(231, self.dfc_dim)
        self.linear2 = nn.Linear(self.dfc_dim, 1)

        # Define batch normalization layers
        self.bn1 = nn.BatchNorm2d(77, eps=1e-05, momentum=0.9, affine=True)
        self.bn2 = nn.BatchNorm1d(self.dfc_dim, eps=1e-05, momentum=0.9, affine=True)

        self.attn2 = Self_Attn( 77,  'relu')

    def forward(self, x, batch_size, pitch_range, attention):

        h0 = F.leaky_relu(self.h0_prev(x), 0.2)
        h1 = self.bn1(self.h1_prev(h0))

        if attention:
          h1, p1 = self.attn2(h1)                                  #([batch_size, 77, 3, 1])) Attention

        h1 = F.leaky_relu(h1, 0.2)
        h1 = h1.view(batch_size, -1)                               #([batch_size, 231])
        h2 = F.leaky_relu(self.bn2(self.linear1(h1)), 0.2)         #[batch_size, 1024]
        h3 = self.linear2(h2)                                      #[batch_size, 1]

        # Apply sigmoid activation to get the probabilities
        h3_sigmoid = torch.sigmoid(h3)

        return h3_sigmoid, h3, h0


# 3. GENERATOR FOR SAMPLING (no batch norm)
class sampling(nn.Module):
    def __init__(self, pitch_range):
        super(sampling, self).__init__()
        self.gf_dim = 64
        self.n_channel = 256

        self.h1 = nn.ConvTranspose2d(in_channels=144, out_channels=pitch_range, kernel_size=(2, 1), stride=(2, 2))
        self.h2 = nn.ConvTranspose2d(in_channels=144, out_channels=pitch_range, kernel_size=(2, 1), stride=(2, 2))
        self.h3 = nn.ConvTranspose2d(in_channels=144, out_channels=pitch_range, kernel_size=(2, 1), stride=(2, 2))
        self.h4 = nn.ConvTranspose2d(in_channels=144, out_channels=1, kernel_size=(1, pitch_range), stride=(1, 2))

        self.h0_prev = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, pitch_range), stride=(1, 2))
        self.h1_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2))
        self.h2_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2))
        self.h3_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2))

        self.linear1 = nn.Linear(100, 1024)
        self.linear2 = nn.Linear(1024, self.gf_dim * 2 * 2)

        self.attn1 = Self_Attn( 144, 'relu')
        self.attn2 = Self_Attn( 144,  'relu')

    def forward(self, z, prev_x, batch_size, pitch_range, attention):
        h0_prev = F.leaky_relu(batch_norm_1d_sampl(self.h0_prev(prev_x)), 0.2)  #[batch_size, 16, 16, 1]
        h1_prev = F.leaky_relu(batch_norm_1d_sampl(self.h1_prev(h0_prev)), 0.2) #[batch_size, 16, 8, 1]
        h2_prev = F.leaky_relu(batch_norm_1d_sampl(self.h2_prev(h1_prev)), 0.2) #[batch_size, 16, 4, 1]
        h3_prev = F.leaky_relu(batch_norm_1d_sampl(self.h3_prev(h2_prev)), 0.2) #[batch_size, 16, 2, 1]

        h0 = F.relu(batch_norm_1d_sampl(self.linear1(z)))                       #(batch_size,1024)
        h1 = F.relu(batch_norm_1d_sampl(self.linear2(h0)))                      #(batch_size,128)
        h1 = h1.view(batch_size, self.gf_dim * 2, 2, 1)                         #(batch_size,128,2,1)

        h1 = conv_prev_concat(h1, h3_prev)                                     #(batch_size,144,2,1)
        h2 = F.relu(batch_norm_1d_sampl(self.h1(h1)))                          #([batch_size, pitch_range, 4, 1])
        h2 = conv_prev_concat(h2, h2_prev)                                     #([batch_size, 144, 4, 1])

        if attention:
          h2,p1 = self.attn1(h2)                                               #([batch_size, 144, 4, 1])  Attention

        h3 = F.relu(batch_norm_1d_sampl(self.h2(h2)))                          #([batch_size, pitch_range, 8, 1])
        h3 = conv_prev_concat(h3, h1_prev)                                     #([batch_size, 144, 8, 1])
        h4 = F.relu(batch_norm_1d_sampl(self.h3(h3)))                          #([batch_size, pitch_range, 16, 1])
        h4 = conv_prev_concat(h4, h0_prev)    				       #([batch_size, 144, 16, 1])

        if attention:
          h4,p2 = self.attn1(h4)                    			       #([batch_size, 144, 16, 1]) Attention

        g_x = torch.sigmoid(self.h4(h4))       				       #([batch_size, 1, 16, 128])
        g_x = one_hot(g_x)

        return g_x
        
         
        
        
def weights_init(m):
    classname = m._class.name_
    if classname.find('Conv') != -1:
        # Initialize convolutional layer weights
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        # Initialize linear layer weights
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        # Initialize batch normalization weights (if needed)
        nn.init.normal_(m.weight.data, 1.0, 0.02)

    # Initialize bias terms if they exist
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)
        
        
        
####################################################################################################################################################
                                                              # PCH metric
###################################################################################################################################################       
        

def kl_divergence(p, q, epsilon):
  ''' 
  Computes kullback-leibler divergence between vectors
  
  Parameters:
     p, q (ndarray) : vectors, representing distribution of pitches
     epsilon (float) : small constant to avoid division by zero
  Returns:
    Kulback-Leibler divergence between p,q
  '''
  # Add epsilon to avoid log(0) and division by zero
  p = np.clip(p, epsilon, 1)
  q = np.clip(q, epsilon, 1)
  return np.sum(kl_div(p, q))


def compute_pch(song):
  '''
  Evaluates the pitch class histogram of a song
  
  Parameters:
    song (ndarray) : song to compute pch
    note_length (int) : length of the note (to weigh frequency of pitches, actually useless if notes all of the same length)
  Returns:
    pch (ndarray) : array with frequency of each pitch (12x1)
  '''
  pch = np.zeros(12)
  for i_bar in range(0,8):
      for i_note in range(0,16):
          note = np.where(song[i_bar,i_note,:]==1)[0]
          pitch_class = note % 12  #remainder, indicates one of the 12 possible pitches
          pch[pitch_class] += 1
  pch /= np.sum(pch)
  return pch
    

def pch_metric(gen_dataset, real_dataset, epsilon, list_real, list_gen):
  '''
  Computes pch for a batch of data
  
  Parameters:
    gen_dataset (ndarray) : generated songs
    real_dataset (ndarray/torch.tensor) : original songs from train or test set
    epsilon (float) : constant for computing kl
    list_real, list_gen (lists) : lists to append pch results
  '''

  for i in range(real_dataset.shape[0]):
    #if (i % 1000)==0:
      #print('Pch evaluated -----> [%d/%d]' % (i, real_dataset.shape[0]))
    if type(real_dataset)==torch.Tensor:
      real_song = real_dataset[i,:,:,:].view(8,16,128).cpu().detach().numpy()
    else:
      real_song = real_dataset[i,:,:,:].reshape(8,16,128)

    gen_song = gen_dataset[i,:,:,:].view(8,16,128).cpu().detach().numpy()

    # Compute metrics
    real_pch = compute_pch(real_song)   #12x1 array
    gen_pch = compute_pch(gen_song)     #12x1 array

    list_real.append(real_pch)
    list_gen.append(gen_pch)

        
        
        
        
        
        
        
####################################################################################################################################################
                                                              # Training and test functions
###################################################################################################################################################
       
        
        
def reduce_mean(x):

    output = torch.mean(x,0, keepdim = True)
    output = torch.mean(output,-1, keepdim = True)
    return output.squeeze()


def reduce_mean_0(x):
    output = torch.mean(x,0, keepdim = True)
    return output.squeeze()


def l2_loss(x,y):
    '''
    Computes MSELoss between vectors, to compute regularization terms of G loss
    Parameters:
      x, y (ndarray) : two vectors
    Returns:
      MSE loss between x,y
    '''
    loss_ = nn.MSELoss(reduction='sum')
    l2_loss_ = loss_(x, y)/2
    return l2_loss_


def random_rotate_song(x, song_idx):
    ''' 
    Rotates song (inverting order of notes)
  
    Parameters:
      x (ndarray) : mini-batch of songs
      song_index (int) : index of the song to rotate
    Returns:
      modified_x (ndarray) : inverted song
    '''
    # Extract the selected song
    selected = x[song_idx,:,:,:].reshape(16*8,128)

    # Circularly rotate the notes within the selected song along the axis of the bars
    rotated = np.flip(selected, axis=0).reshape(8, 16, 128)

    modified_x = x
    modified_x[song_idx,:,:,:] = rotated

    return (modified_x)
    
    
    
### TRAINING ###
    
def training(epochs, lrD, lrG, p_invert, batch_size, nz, lambda1, lambda2, train_data, prev_train_data, attention, device, epsilon, pch):
  ''' 
  Training function
  
  Parameters:
     epochs (int) :     number of training epochs
     lrD, lrG (int) :   learning rates for discriminator and generator optimizers
     p_invert (float) : probability of inverting a song (apply random_rotate_song)
     batch_size (int) : size of minibatches
     nz (int) :         size of noise vector for G
     lambda1, lambda2 (float) : parameters for the two regularization terms in the G loss
     train_data, prev_train_data (ndarray) : train dataset and prev_train dataset 
     attention (bool):  determines if self attention layers are applied to G and D
     device :           device
     epsilon (float) :  constant for computing kl divergence (if pch=True)
     pch (bool)      :  determines if pitch class histograms are computed and kl divergence evaluated
  '''

  pitch_range = 128

  print('Starting training \n')

  dataset = train_data
  prev_dataset = prev_train_data

  netG = generator(pitch_range).to(device)
  netD = discriminator(pitch_range).to(device)

  netD.train()  # prepare batch norm layers for training
  netG.train()
  optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(0.5, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999))

  # Define learning rate schedulers
  schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=10, gamma=0.5)
  schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=10, gamma=0.5)

  fixed_noise = torch.randn(batch_size, nz, device=device)
  real_label = 0.9  # one sided label smoothing
  fake_label = 0
  average_lossD = 0
  average_lossG = 0
  average_D_x = 0
  average_D_G_z = 0

  lossD_list = []
  lossD_list_all = []
  lossG_list = []
  lossG_list_all = []
  D_x_list = []
  D_G_z_list = []

  kl_list = [] 
  real_pch_list = []
  gen_pch_list = [] 

  batch_songs = int(batch_size / 8)  # songs per batch: 9 if batch_size=72, 4 if batch_size=32
  data_max = int(dataset.shape[0] / batch_songs)  # max division of train dataset by batch_songs

  for epoch in range(epochs):

      torch.cuda.empty_cache()  # empty cache for memory

      list_real_pch = []
      list_gen_pch = []

      sum_lossD = 0
      sum_lossG = 0
      sum_D_x = 0
      sum_D_G_z = 0
      for d_count in range(0, data_max + 1, batch_songs):  # step

          ##########################
          # (0) Load Minibatches of data
          #########################

          curr_x = dataset[(d_count):(batch_songs + d_count), :, :, :]
          prev_x = prev_dataset[(d_count):(batch_songs + d_count), :, :, :]

          #Random inversion
          rn = np.random.uniform()
          if rn < p_invert:
            song_idx = random.randint(0, curr_x.shape[0] - 1)
            curr_x = random_rotate_song(curr_x, song_idx)
            prev_x = random_rotate_song(prev_x, song_idx)

          prev_x = torch.from_numpy(prev_x).view(batch_size, 1, 16, 128).to(device).float()
          curr_x = torch.from_numpy(curr_x).view(batch_size, 1, 16, 128).to(device).float()

          criterion = nn.BCEWithLogitsLoss()

          ############################
          # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
          ###########################
          netD.zero_grad()

          D, D_logits, fm = netD(curr_x, batch_size, pitch_range, attention)
          d_loss_real = reduce_mean(criterion(D_logits, 0.9 * torch.ones_like(D)))
          d_loss_real.backward(retain_graph=True)
          D_x = D.mean().item()
          sum_D_x += D_x

          noise = torch.randn(batch_size, nz, device=device)
          fake = netG(noise, prev_x, batch_size, pitch_range, attention)
          D_, D_logits_, fm_ = netD(fake.detach(), batch_size, pitch_range,attention)
          d_loss_fake = reduce_mean(criterion(D_logits_, torch.zeros_like(D_)))
          d_loss_fake.backward(retain_graph=True)
          D_G_z1 = D_.mean().item()
          errD = d_loss_real + d_loss_fake
          errD = errD.item()
          lossD_list_all.append(errD)
          sum_lossD += errD
          optimizerD.step()

          ############################
          # (2) Update G network: maximize log(D(G(z)))
          ###########################
          netG.zero_grad()

          D_, D_logits_, fm_ = netD(fake.detach(), batch_size, pitch_range, attention)
          D, D_logits, fm = netD(curr_x, batch_size, pitch_range, attention)

          g_loss0 = reduce_mean(criterion(D_logits_, torch.ones_like(D_)))
          features_from_g = reduce_mean_0(fm_)
          features_from_i = reduce_mean_0(fm)
          fm_g_loss1 = torch.mul(l2_loss(features_from_g, features_from_i), lambda2)

          mean_image_from_g = reduce_mean_0(fake)
          smean_image_from_i = reduce_mean_0(curr_x)
          fm_g_loss2 = torch.mul(l2_loss(mean_image_from_g, smean_image_from_i), lambda1)

          errG = g_loss0 + fm_g_loss1 + fm_g_loss2
          errG.backward(retain_graph=True)
          D_G_z2 = D_.mean().item()
          optimizerG.step()

          ############################
          # (3) Update G network again: maximize log(D(G(z)))
          ###########################
          netG.zero_grad()

          fake_ = netG(noise, prev_x, batch_size, pitch_range, attention)
          D_, D_logits_, fm_ = netD(fake_.detach(), batch_size, pitch_range, attention)
          D, D_logits, fm = netD(curr_x, batch_size, pitch_range, attention)

          g_loss0 = reduce_mean(criterion(D_logits_, torch.ones_like(D_)))
          features_from_g = reduce_mean_0(fm_)
          features_from_i = reduce_mean_0(fm)
          fm_g_loss1 = torch.mul(l2_loss(features_from_g, features_from_i), lambda2)

          mean_image_from_g = reduce_mean_0(fake_)
          smean_image_from_i = reduce_mean_0(curr_x)
          fm_g_loss2 = torch.mul(l2_loss(mean_image_from_g, smean_image_from_i), lambda1)

          errG = g_loss0 + fm_g_loss1 + fm_g_loss2
          errG_item = errG.item()
          sum_lossG += errG_item
          errG.backward(retain_graph=True)
          lossG_list_all.append(errG.item())

          D_G_z2 = D_.mean().item()
          sum_D_G_z += D_G_z2
          optimizerG.step()

          if pch:
            # append pch for the minibatch
            gen_song = fake_.view(batch_songs,8,16,128)
            real_song = curr_x.view(batch_songs,8,16,128)
            pch_metric(gen_song, real_song, epsilon, list_real_pch, list_gen_pch)



      # Step the learning rate schedulers
      schedulerD.step()
      schedulerG.step()

      if pch:
        avg_real_pch = np.mean(list_real_pch, axis=0)
        avg_gen_pch = np.mean(list_gen_pch, axis=0)
        kl = kl_divergence(avg_real_pch, avg_gen_pch, epsilon)
      else:
        avg_real_pch = None
        avg_gen_pch = None
        kl = None

      if epoch % 2 == 0:
          print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Kullback-Leibler: %.4f'
                % (epoch, epochs, errD, errG, D_x, D_G_z1, D_G_z2, kl))

      average_lossD = (sum_lossD / int(dataset.shape[0]))
      average_lossG = (sum_lossG / int(dataset.shape[0]))
      average_D_x = (sum_D_x / int(dataset.shape[0]))
      average_D_G_z = (sum_D_G_z / int(dataset.shape[0]))

      lossD_list.append(average_lossD)
      lossG_list.append(average_lossG)
      D_x_list.append(average_D_x)
      D_G_z_list.append(average_D_G_z)
      
      kl_list.append(kl)
      real_pch_list.append(avg_real_pch)
      gen_pch_list.append(avg_gen_pch)
      

      #print('==> Epoch: {} Average lossD: {:.10f} average_lossG: {:.10f},average D(x): {:.10f},average D(G(z)): {:.10f} '.format(
        #epoch, average_lossD, average_lossG, average_D_x, average_D_G_z))

      np.save('lossD_list.npy', lossD_list)
      np.save('lossG_list.npy', lossG_list)
      np.save('lossD_list_all.npy', lossD_list_all)
      np.save('lossG_list_all.npy', lossG_list_all)
      np.save('D_x_list.npy', D_x_list)
      np.save('D_G_z_list.npy', D_G_z_list)

      # Save pch results
      np.save('kl_train.npy',kl_list)
      np.save('real_pch_train.npy', real_pch_list)
      np.save('gen_pch_train.npy', gen_pch_list)

      # do checkpointing
      torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('../models', epoch))
      torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('../models', epoch))
      
      
 
      
 ### TESTING  ###      
def testing(nz, test_data, state_dict, attention, device, epsilon, pch):
  ''' 
  Testing function: conditioned on the first bar of each test melody, the generator generates a new melody
  
  Parameters:
     nz (int) : size of noise vector for G
     test_data: test dataset and prev_train dataset 
     state_dict : state of the Generator after training
     attention (bool): determines if self attention layers are applied to G
     device : device
     epsilon (float) :  constant for computing kl divergence (if pch=True)
     pch (bool)      :  determines if pitch class histograms are computed and kl divergence evaluated
  '''

  pitch_range = 128

  print('Starting testing \n')

  dataset = torch.from_numpy(test_data).to(device).float()

  batch_size = 8
  n_bars = 7

  list_real_pch = []
  list_gen_pch = []

  netG_sampling = sampling(pitch_range).to(device)

  # Filter out batch normalization parameters
  filtered_state_dict = {k: v for k, v in state_dict.items() if 'bn1d' not in k and 'bn2d' not in k}
  netG_sampling.load_state_dict(filtered_state_dict, strict=False)

  output_songs = []

  batch_songs = int(batch_size / 8)  # songs per batch: one at a time for testing
  data_max = int(dataset.shape[0] / batch_songs)  # max division of train dataset by batch_songs

  # For each song in the test dataset, generate a melody starting from the first bar
  for d_count in range(0, data_max, batch_songs):  # step

      if (d_count % 1000)==0 or (d_count==data_max):
        print('Songs generated -----> [%d/%d]' % (d_count, data_max))

      curr_x = dataset[(d_count):(batch_songs + d_count), :, :, :]
      list_song = []

      seed_bar = curr_x[0, 0, :, :]
      first_bar = seed_bar.view(1,1,16,128)

      list_song.append(first_bar)
      noise = torch.randn(batch_size, nz, device=device).to(device).float() # z, shape torch.Size([8, 100])

      for bar in range(n_bars):
          z = noise[bar].view(1,nz)
          if bar == 0:
              prev = seed_bar.view(1,1,16,128)
          else:
              prev = list_song[bar-1].view(1,1,16,128)
          sample = netG_sampling(z, prev,1,pitch_range, attention)
          list_song.append(sample)

      # Turn song (list of bars) into tensor
      gen_song = torch.stack([sample for sample in list_song], dim=0)
      gen_song = gen_song.view(1,8,16,128)
      # Compute pch
      real_song = curr_x.view(1,8,16,128)

      output_songs.append(gen_song)
      # Evaluate pch
      if pch:
        pch_metric(gen_song, real_song, epsilon, list_real_pch, list_gen_pch)

  if pch:    
    avg_real_pch = np.mean(list_real_pch, axis=0)
    avg_gen_pch = np.mean(list_gen_pch, axis=0)
    kl = kl_divergence(avg_real_pch, avg_gen_pch, epsilon)
    np.save('real_pch_test.npy', avg_real_pch)
    np.save('gen_pch_test.npy', avg_gen_pch)
    np.save('kl_test.npy', kl)

  
  print('Testing completed, songs created: {}'.format(len(output_songs)))
  print('KL divergence for generated songs on test set: {}'.format(kl))
  return(output_songs, kl)

  
  
  
  
  
####################################################################################################################################################
                                                              # Generation of songs
################################################################################################################################################### 
  

  
#Function to generate one single sample of generated song, starting from a random bar in the test set

def generate_sample(nz, test_data, state_dict, attention, device):
  ''' 
  The one song version of testing: generates one song from a bar of a random test song
  
  Parameters:
     nz (int) : size of noise vector for G
     test_data: test dataset and prev_train dataset 
     state_dict : state of the Generator after training
     attention (bool): determines if self attention layers are applied to G
     device : device
  '''
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  batch_size = 8
  n_bars = 7
  pitch_range = 128

  netG_sampling = sampling(pitch_range).to(device)

  idx_song = np.random.randint(0, test_data.shape[0])
  idx_bar = np.random.randint(0, test_data.shape[1])

  dataset = torch.from_numpy(test_data).to(device).float()
  seed_bar = dataset[idx_song, idx_bar, :, :]

  # Filter out batch normalization parameters
  filtered_state_dict = {k: v for k, v in state_dict.items() if 'bn1d' not in k and 'bn2d' not in k}
  netG_sampling.load_state_dict(filtered_state_dict, strict=False)

  output_song = []
  list_song = []

  first_bar = seed_bar.view(1,1,16,128)
  list_song.append(first_bar)
  noise = torch.randn(batch_size, nz, device=device).to(device).float() # z, shape torch.Size([8, 100])

  for bar in range(n_bars):
      z = noise[bar].view(1,nz)
      if bar == 0:
          prev = seed_bar.view(1,1,16,128)
      else:
          prev = list_song[bar-1].view(1,1,16,128)
      sample = netG_sampling(z, prev,1,pitch_range, attention)
      list_song.append(sample)
      
  gen_song = torch.stack([sample for sample in list_song], dim=0).view(8,16,128)
  
  return gen_song
  
  
  
def tensor_to_midi(tensor, output_path):
  ''' 
  Turns generated song into midi file
  
  Parameters:
     tensor (torch.Tensor) : generated song
     output_path (str) : path to save output midi file
  Returns:
    track : midi track
  '''
  # Create a new MIDI file with one track, from (8,16,128) tensor
  mid = MidiFile()
  track = MidiTrack()
  mid.tracks.append(track)

  # Set the tempo (optional, default is 500000)
  tempo = 500000
  note_length = 96  # Assuming each timestep represents 24 ticks
  velocity = 100    # Default velocity for notes

  for i, bar in enumerate(tensor):
      for j, timestep in enumerate(bar):
          #Identify active notes
          notes = np.where(timestep !=0)[0]
          if len(notes) >= 0:
              for note in notes:
                  track.append(Message('note_on', note=int(note), velocity=velocity, time=0))
                  track.append(Message('note_off', note=int(note), velocity=velocity, time=note_length))
          else:
              track.append(Message('note_off', note=0, velocity=0, time=note_length))

  # Save MIDI
  mid.save(output_path)
  print(f"MIDI file saved to {output_path}")
  return track

