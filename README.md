# GAN-music-generator-NNDL


%<h2 align="center">
  %<img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExdXYyeGl3MzR3aWJydjk4N3dhbXU4anViaXFvOTh4ODlxYjA1aHJ1eSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tqfS3mgQU28ko/giphy.gif", width="250">
%</h2>


An implementation of a Music generator through Generative Adversarial network, for the final project of the course Neural Networks and Deep Learning of the Master's Degree in Physics of Data at University of Padua. 

In this project, we explore the use of Convolutional Neural Networks (CNNs) within a Generative Adversarial Network (GAN) framework to generate sequences of MIDI notes one bar at a time.
The base model is inspired by     , but we propose the addition of a self-attention mechanism in the GAN architecture, which allows attention-driven, long-range dependency modeling of the song bars, enabling the model to capture more complexrelationships within the music.
We also propose two quantitative metrics for melody quality evaluation, one focused on the note frequency distribution, the other, more general, leverages embeddings from a pre-trained autoencoder. Through this set
of metrics we are able to evaluate the impact of self-attention layers in the training phase, showing an increase in similarity between real and generated data.    

The repository contains:
- The main jupyter notebook, with the full code implementation;
- Three python libraries we implemented, to be used in the main notebook: 

    - preprocess_midi.py : is the library containing all the functions needed for the preprocessing of MIDI files; 
    - gan_model.py : is the library containing everything about the GAN model, from the architecture, to the training and testing function, the functions to generate a new song and convert it into a MIDI file, and the pch metric functions.
    - autoenc_model.py : is the library containing the autoencoder model, and its training function, used to implement the clustering and reconstruction loss.



# (NEW) Excel human evaluation + link to the plot
https://colab.research.google.com/drive/13Xd4dlbJPiKC-uwzge-IZ0ghvE_di7ut?usp=sharing

https://docs.google.com/spreadsheets/d/1riGOAVK5oGD6kHRmNW_9Ac8J0PnNvr05ityVzDi17eM/edit?usp=sharing

# Colab
https://colab.research.google.com/drive/1ZkgJVSV-4dyn1gbTx5fi6Sq11rhaVGTW?authuser=0#scrollTo=IDLgv01-u9E7
diagramma -->  https://docs.google.com/drawings/d/164SROfcnHE3tQ2bffMIW8DdHqYKbzCQ1ZhQqXNy6rtc/edit?usp=sharing

--- 

Autoencoder: https://colab.research.google.com/drive/1tk_-86cuKZ4Ivxn56XGBGsZQ67eylMC9?usp=sharing

---

# Overleaf report
https://www.overleaf.com/1897935423tbfwqsgqbtfy#f01106
