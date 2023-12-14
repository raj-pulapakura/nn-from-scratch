# nn-from-scratch

**Neural Network from Scratch with NumPy, to solve the MNIST classification problem.**

[![Screenshot 2023-12-14 161047](https://github.com/raj-pulapakura/nn-from-scratch/assets/87762282/0ff2031a-ed99-41ef-a884-539f95298c39)](https://www.youtube.com/watch?v=OC_yDwBdIJQ)

# 🤗 Test drive

## Clone the repository

```bash
git clone https://github.com/raj-pulapakura/nn-from-scratch.git
```

## Install packages

```
cd nn-from-scratch
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

1. Change directory to project
2. Create virtual environment. This ensures that the project does not affect your global packages.
3. Activate virtual environment.
4. Install required packages.

## Train the model

```
python main.py
```

# About

## Neural Networks

Neural Networks have revolutionised the AI world because of their predictive power and scalability. Neural Network are the fundamental engine of more complicated architectures such as Convolutional Neural Networks, Recurrent Neural Networks, and Transformers.

## MNIST

MNIST, arguably the most famous dataset in the AI world, is a repository of hand-written digits.

![mnist dataset](https://github.com/raj-pulapakura/nn-from-scratch/assets/87762282/00099558-f245-4b1d-9f31-37c0e73f79b8)

Source: https://paperswithcode.com/dataset/mnist

MNIST served as a benchmark for machine learning models to evaulate their performance. However, as the deep learning world has matured and models have become increasingly more powerful, solving the MNIST dataset has become a trivial task. Therefore, more comprehensive datasets such as ImageNet (for vision models) have emerged.

Still, training neural networks on the MNIST dataset is a great exercise for beginners. Due to its simplicity, it's a great tool to expose novices to the model development process.

## Purpose of this repository

ML and DL libraries such as TensorFlow and PyTorch abstract away the computational complexity of models. This allows developers to iterate quickly to build models that solve problems.

However, this abstraction means that we often neglect what's happening under the hood. This is good for quickly and developing and testing models - it would be extremely painful if we had to implement back-propagation every time we created a model 😭.

Still, implementing a neural network from scratch is an insightful and eye-opening exercise. I believe all machine learning/deep learning engineers should implement back-propagation at least once in their life. It's a great exercise for strengthening your understanding of the fundamental algorithms powering neural networks.

Now, once you've implemented back-propagation once, you probably don't need to do it ever again. For me, the sole point of implementing a neural network from scratch was to prove to myself that I actually _understood neural networks_.

This repository is the result of that. In this repository, I build neural networks from the ground up, implementing all the fundamental equations myself. No TensorFlow. No PyTorch. Pure Python and NumPy.

My hope is that you view this repository and are inspired to implement a neural network yourself - its a wonderful journey.

# Your next steps

I highly recommend you clone this repository and tinker with the code. Take some time to analyse the code, fiddle with the hyperparameters, and see if you can get more than 95% accuracy.

I've also made it really easy for you to add more layers to the network, and even implement your own custom layers. Remember, experimentation is the only way you get results.

Have a great time ✌️
