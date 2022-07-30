<p align="center"><img src="assets/logo.png" width="480"\></p>

**This repository has gone stale as I unfortunately do not have the time to maintain it anymore. If you would like to continue the development of it as a collaborator send me an email at eriklindernoren@gmail.com.**

## PyTorch-GAN
Collection of PyTorch implementations of Generative Adversarial Network varieties presented in research papers. Model architectures will not always mirror the ones proposed in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. Contributions and suggestions of GANs to implement are very welcomed.

<b>See also:</b> [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)

## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Deep Convolutional GAN](#deep-convolutional-gan)
    + [GAN](#gan)
    + [Wasserstein GAN GP](#wasserstein-gan-gp)

## Installation
    $ git clone https://github.com/eriklindernoren/PyTorch-GAN
    $ cd PyTorch-GAN/
    $ sudo pip3 install -r requirements.txt

## Implementations   

### Deep Convolutional GAN
_Deep Convolutional Generative Adversarial Network_

#### Authors
Alec Radford, Luke Metz, Soumith Chintala

#### Abstract
In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

[[Paper]](https://arxiv.org/abs/1511.06434) [[Code]](implementations/dcgan/dcgan.py)

#### Run Example
```
$ cd implementations/dcgan/
$ python3 dcgan.py
```

<p align="center">
    <img src="assets/dcgan.gif" width="240"\>
</p>

### GAN
_Generative Adversarial Network_

#### Authors
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

#### Abstract
We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

[[Paper]](https://arxiv.org/abs/1406.2661) [[Code]](implementations/gan/gan.py)

#### Run Example
```
$ cd implementations/gan/
$ python3 gan.py
```

<p align="center">
    <img src="assets/gan.gif" width="240"\>
</p>

### Wasserstein GAN GP
_Improved Training of Wasserstein GANs_

#### Authors
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville

#### Abstract
Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only low-quality samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models over discrete data. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

[[Paper]](https://arxiv.org/abs/1704.00028) [[Code]](implementations/wgan_gp/wgan_gp.py)

#### Run Example
```
$ cd implementations/wgan_gp/
$ python3 wgan_gp.py
```

<p align="center">
    <img src="assets/wgan_gp.gif" width="240"\>
</p>

