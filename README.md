# GAN-music-generator-NNDL


<h2 align="center">
  <img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExdXYyeGl3MzR3aWJydjk4N3dhbXU4anViaXFvOTh4ODlxYjA1aHJ1eSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tqfS3mgQU28ko/giphy.gif", width="250">
</h2>


An implementation of a Music generator through Generative Adversarial network, for the final project of the course Neural Networks and Deep Learning of the Master's Degree in Physics of Data at University of Padua.    

# To-Do List
- [x] Self attention mechanism (https://github.com/heykeetae/Self-Attention-GAN + https://arxiv.org/pdf/1805.08318)
- [x] Implement the evaluation metric 1 + 2
- [x] add the one hot encoding at the end of the G
- [x] add inversion of a song with certain prob
- [ ] tune hyperparameters: lr, batch sizes, number of D vs G updates in training, and network architectures 
    
- [ ] Spectral Normalization: Normalize the weights of the discriminator using spectral normalization to enforce the Lipschitz constraint.

# Colab
con adattamento a 16 beats per bar: https://colab.research.google.com/drive/1EYGU1iSsgXA7P88fKza-njO8uE4TvzVL?usp=sharing

con VAE 
https://colab.research.google.com/drive/1iwV3B2Ad98STDmjJPjHhACAalj21I_Jo?usp=sharing

VAE aknowlegements
https://github.com/search?q=repo%3ASashaMalysheva/Pytorch-VAE%20name&type=code

Per Jacopo : https://colab.research.google.com/drive/1lo0VK5TcagdNWtC6xFh-A3F83jNjbKey?usp=sharing

**COMPLETE with libraries (autoenc_model.py, gan_model.py)** : https://colab.research.google.com/drive/1ZkgJVSV-4dyn1gbTx5fi6Sq11rhaVGTW?authuser=0#scrollTo=IDLgv01-u9E7

---

last version : https://colab.research.google.com/drive/1azyECFzRPYUf0Ex_zrMQx1oll2eQKxU4?usp=sharing        

reorganized version with functions: https://colab.research.google.com/drive/1GvLjvIDQEiXQOgar9xK86bMKjOgfCbmW?usp=sharing&authuser=1#scrollTo=wtQLW2b-u9E8    

--- 

autoencoder work in progress: https://colab.research.google.com/drive/1tk_-86cuKZ4Ivxn56XGBGsZQ67eylMC9?usp=sharing
- [x] add noise
- [x] recontruction loss function
- [ ] try on data of the GAN and on the real samples + clustered img


# Overleaf report
https://www.overleaf.com/1897935423tbfwqsgqbtfy#f01106
