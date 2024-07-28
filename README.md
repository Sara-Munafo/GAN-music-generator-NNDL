# GAN-music-generator-NNDL


<h2 align="center">
  <img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExdXYyeGl3MzR3aWJydjk4N3dhbXU4anViaXFvOTh4ODlxYjA1aHJ1eSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tqfS3mgQU28ko/giphy.gif", width="250">
</h2>


An implementation of a Music generator through Generative Adversarial network, for the final project of the course Neural Networks and Deep Learning of the Master's Degree in Physics of Data at University of Padua.    

# To-Do List
- [ ] Change lrD and lrG and keep track of the tests
- [ ] Print all the loss values and tune feature matching params
- [ ] Implement the 3 evaluation metrics
- [ ] Self attention mechanism (https://github.com/heykeetae/Self-Attention-GAN)
- [ ] tune hyperparameters: batch sizes, number of D vs G updates in training, and network architectures
- [ ] Avoid Mode Collapse: Use techniques like unrolled GANs to address mode collapse
- [ ] Spectral Normalization: Normalize the weights of the discriminator using spectral normalization to enforce the Lipschitz constraint.
- [ ] Historical Averaging: Penalize the network parameters by the distance from historical averages
- [ ] Minimize Wasserstein Loss: Wasserstein GANs (WGANs) use a different loss function that provides better gradients for training.

# Colab
con adattamento a 16 beats per bar: https://colab.research.google.com/drive/1EYGU1iSsgXA7P88fKza-njO8uE4TvzVL?usp=sharing

con VAE 
https://colab.research.google.com/drive/1iwV3B2Ad98STDmjJPjHhACAalj21I_Jo?usp=sharing

VAE aknowlegements
https://github.com/search?q=repo%3ASashaMalysheva/Pytorch-VAE%20name&type=code

Con Augmentation    
https://colab.research.google.com/drive/1T9ps_vxQM4YXZTjN4wCv5uT3_qOHgutE?usp=sharing

sistemato con sampling: https://colab.research.google.com/drive/1jHcNqrZMhFjROF0IHkW2erDdzCnXcM3L?usp=sharing

last version with lr : https://colab.research.google.com/drive/1azyECFzRPYUf0Ex_zrMQx1oll2eQKxU4?usp=sharing    

tentativi fallimentari : https://colab.research.google.com/drive/1BjWN_4Um4xi4yT7G79zSWt08TxSHvJ8y?usp=sharing

# Overleaf report
https://www.overleaf.com/1897935423tbfwqsgqbtfy#f01106
