# GradNet

Training and testing of our GradNet structure. Gradnet.ipynb contains the code for testing our pre-trained GradNet with a pre-trained CNN, and for training your own.

### About GradNet:

In our experiments we test the hypothesis that for a given deep neural network pre-trained for a classification problem, a quadratic separator on the gradient space can outperform the original ("base") network. We wish to find the proper separator by training a two-layer NN called GradNet. In order to avoid having too much parameters to train, we chose the hidden layer of our network to have a block-like structure demonstrated in Figure 1. This model is capable of capturing connections between gradients from adjacent layers of the base network.

<p align="center"> 
  <img src="http://info.ilab.sztaki.hu/~alexievr/GradNet.png" width="500">
</p>
<p align="center">Figure 1.<p align="center">


In order to find the optimal architecture, the right normalization process and regularization technique, and the best optimization method, we experimented with a large number of setups. We measured the performance of these various kinds of GradNet models on the gradient space of a CNN trained on the first half of the CIFAR-10 training dataset. We used the other half of the dataset and random labels to generate gradient vectors to be used as training input for the GradNet. In the testing phase we use all of the gradient vectors for every data point in the test set, we give them all to the network as inputs, and we define the prediction as the index of the maximal element in the sum of the outputs. During our experiments, as a starting point we stopped the underlying original CNN at 0.72 accuracy and compared numerous different settings.

Learning curves for the different networks are presented here.

| <img src="http://info.ilab.sztaki.hu/~alexievr/optimalizacio.png" width="400"> | <img src="http://info.ilab.sztaki.hu/~alexievr/regularizacio.png" width="400"> | 
|:---:|:---:|
| Optimization methods | Regularization methods |

| <img src="http://info.ilab.sztaki.hu/~alexievr/percent.png" width="400"> | <img src="http://info.ilab.sztaki.hu/~alexievr/struktura.png" width="400"> | 
|:---:|:---:|
| Selection percentile | Structure |

| <img src="http://info.ilab.sztaki.hu/~alexievr/normalizacio_5_25_10.png" width="400"> | <img src="http://info.ilab.sztaki.hu/~alexievr/normalizacio_5_100_25.png" width="400"> | 
|:---:|:---:|
| Normalization with structure 5+25+10 | Normalization with structure 5+100+25 |


To show the performance of the GradNet with the best settings, we took snapshots of a CNN at progressively increasing levels of pre-training, and we trained the GradNet on the gradient sets of these networks. We ran these tests using a CNN trained on half of the CIFAR dataset and with one trained on half of MNIST. Table 1 shows the accuracies of all the base networks together with the accuracies of the corresponding GradNets.

<p align="center"> 
  <img src="http://info.ilab.sztaki.hu/~alexievr/performance.png" width="500">
</p>
<p align="center">Table 1.<p align="center">
