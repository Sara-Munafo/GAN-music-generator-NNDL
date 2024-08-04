import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.autograd import Variable
from tqdm import tqdm
import random

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

############################################################################################################################################
                                               # Autoencoder model
############################################################################################################################################                                               

class AutoEncoder(nn.Module):
    def __init__(self, batch_size, channel_num, feature_size_h, feature_size_w, z_size, pitch_range):
        super(AutoEncoder, self).__init__()

        self.batch_size = batch_size
        self.channel_num = channel_num  # 1
        self.feature_size_h = feature_size_h  # 16
        self.feature_size_w = feature_size_w  # 128
        self.z_size = z_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, pitch_range), stride=(1, 2)),  # (batch_size, 16, 16, 1)
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2)),  # (batch_size, 16, 8, 1)
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2)),  # (batch_size, 16, 4, 1)
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2)),  # (batch_size, 16, 2, 1)
            nn.LeakyReLU(0.2)
        )
        self.linear1 = nn.Linear(32, self.z_size)
        self.linear2 = nn.Linear(self.z_size, 32)


        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2)),  # (batch_size, 16, 4, 1)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2)),  # (batch_size, 16, 8, 1)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2)),  # (batch_size, 16, 16, 1)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, pitch_range), stride=(1, 2)),  # (batch_size, 1, 16, 128)
            nn.Sigmoid()  # Output should be in the range [0, 1]
        )

    def forward(self, x):
        # Encode x
        encoded = self.encoder(x)

        # Flatten the encoded feature map
        encoded_flat = encoded.view(self.batch_size, -1)  # (72, 32)

        # Projection
        z_latent= self.linear1(encoded_flat)              # (72,z_size)
        #print('z_latent',z_latent.shape)

        # Reshape for decoding
        decoded_input = self.linear2(z_latent)            # (72, 32)
        decoded_input = decoded_input.view(self.batch_size, self.channel_num, self.feature_size_h, self.feature_size_w)  # (batch_size, 1, 16, 128)
        #print('decoded_input', decoded_input.shape)

        # Decode back to original dimensions
        x_reconstructed = self.decoder(decoded_input)

        return z_latent, x_reconstructed
        
 
 
 
############################################################################################################################################
                                               # Training
############################################################################################################################################        
        
        
# Function to randomly rotate a song during training        
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
    
    
    
    
# Training function    

def train_autoencoder(model, epochs, learning_rate, batch_size, train_data, device, state_path):
  ''' 
  Trains autoencoder model
  
    Parameters:
      model (nn.Module) : initialized model to train
      epochs (int) : number of training epochs
      learning_rate (float) : learning rate of the optimizer
      batch_size (int) : batch_size
      train_data (ndarray) : train dataset
      device : device
      state_path (str) : path to save the final state of the model
  '''
  # Initialize model, loss and optimizer
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  batch_songs = int(batch_size / 8)  # songs per batch: 9 if batch_size=72, 4 if batch_size=32
  data_max = int(train_data.shape[0] / batch_songs)  # max division of train dataset by batch_songs


  #training
  for epoch in range(epochs):
      torch.cuda.empty_cache()  # empty cache for memory
      model.train()
      running_loss = 0.0
      for d_count in range(0, data_max + 1, batch_songs):  # step
          x = train_data[(d_count):(batch_songs + d_count), :, :, :]

          rn = np.random.uniform()
          if rn < 0.5:
            song_idx = random.randint(0, x.shape[0] - 1)
            x = random_rotate_song(x, song_idx)

          x = torch.from_numpy(x).view(batch_size, 1, 16, 128).to(device).float()
          optimizer.zero_grad()

          #  forward
          _, outputs = model(x) # return z_latent, x_reconstructed
          loss = criterion(outputs, x)

          #  backward e ottimizzazione
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

      # Stampa la loss media per epoca
      print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/data_max:.4f}')

  # Salva il modello
  torch.save(model.state_dict(), state_path)
  
  
  
  
############################################################################################################################################
                                               # Metrics of similarity
############################################################################################################################################        
  
  
def get_latent_vectors(dataset, batch_size, model, device):
  ''' 
  Gets latent representation and encoded representation
  
    Parameters:
      dataset (ndarray) : dataset to encode
      batch_size (int) : batch_size
      device : device
      model (nn.module) : trained model
    Returns:
      latent_vectors (ndarray) : latent representation of dataset
      encoded_vectors *ndarray) : encoded dataset
    
  '''

  latent_vectors = []
  encoded_vectors = []

  batch_songs = int(batch_size / 8)  # songs per batch: 9 if batch_size=72, 4 if batch_size=32
  data_max = int(dataset.shape[0] / batch_songs)  # max division of train dataset by batch_songs

  with torch.no_grad():  # No need to compute gradients for this
      for d_count in range(0, data_max + 1, batch_songs):  # step
          x = dataset[(d_count):(batch_songs + d_count), :, :, :]
          x = torch.from_numpy(x).view(batch_size, 1, 16, 128).to(device).float()
          z, encoded = model(x)
          latent_vectors.append(z.cpu().numpy())
          encoded_vectors.append(encoded.cpu().numpy())

  latent_vectors = np.concatenate(latent_vectors, axis=0)
  encoded_vectors = np.concatenate(encoded_vectors, axis = 0)
  return(latent_vectors, encoded)




def pca_cluster(latent_vectors, n_comp, num_clust):
  ''' 
  Performs pca and clustering of the latent representations
  
    Parameters:
      latent_vectors (ndarray) : latent representation of dataset
      n_comp (int) : number of components for the PCA
      num_clust (int) : number of clusters for Kmeans
    Returns:
      pca_result (ndarray) : latent vectors after PCA
      clusters (ndarray) : clusters after KMeans
  '''
  # Perform PCA to reduce to 2 dimensions
  pca = PCA(n_components=n_comp)
  pca_result = pca.fit_transform(latent_vectors)

  kmeans = KMeans(n_clusters=num_clust)
  clusters = kmeans.fit_predict(pca_result)

  return(pca_result,clusters)




def compute_reconstruction_loss(x, x_reconstructed, device):
  ''' 
  Computes reconstruction error
  
    Parameters:
      x (ndarray/torch.Tensor) : vectors of the original dataset
      x_reconstructed (ndarray/torch.Tensor) : reconstructed vectors from the decoder
      device
    Returns:
      loss.item() : reconstruction loss
  '''
  # Check if the inputs are numpy and convert them to torch arrays if necessary
  if isinstance(x, np.ndarray):
      x = torch.from_numpy(x).float().to(device)
  if isinstance(x_reconstructed, np.ndarray):
      x_reconstructed = torch.from_numpy(x_reconstructed).float().to(device)

  # Ensure that both inputs are torch tensors
  assert isinstance(x, torch.Tensor), "x should be a torch tensor"
  assert isinstance(x_reconstructed, torch.Tensor), "x_reconstructed should be a torch tensor"

  # Check for dimensions: reconstruction happens in batches of batch_size, not all dataset gets processed
  tot_songs = x_reconstructed.shape[0]
  songs_perbatch = int(tot_songs/8)
  x = x[:songs_perbatch,:,:,:].view(tot_songs,1,16,128)
  # Compute the reconstruction loss
  criterion = nn.MSELoss()
  loss = criterion(x, x_reconstructed)

  return loss.item()
