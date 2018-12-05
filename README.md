# GradNet

Training and testing of our GradNet structure. Gradnet.ipynb contains the code for testing our pre-trained GradNet with a pre-trained CNN, and for training your own.

### About GradNet:

We assess the expressive power of the outer product manifold by inspecting the gradient-sets of a CNN. In our experiments we test the hypothesis that for a given deep neural network pre-trained for a classification problem, a quadratic separator on the gradient space can outperform the original ("base") network. We wish to find the proper separator by training a two-layer NN called GradNet. In order to avoid having too much parameters to train, we chose the hidden layer of our network to have a block-like structure demonstrated below. This model is capable of capturing connections between gradients from adjacent layers of the base network.

<img src="http://info.ilab.sztaki.hu/~alexievr/Minornet_abra_2.png" width="500">

In order to find the optimal architecture, the right normalization process and regularization technique, and the best optimization method, we experimented with a large number of setups. We measured the performance of these various kinds of GradNet models on the gradient space of a CNN trained on the first half of the CIFAR-10 training dataset. We used the other half of the dataset and random labels to generate gradient vectors to be used as training input for the GradNet. In the testing phase we use all of the gradient vectors for every data point in the test set, we give them all to the network as inputs, and we define the prediction as the index of the maximal element in the sum of the outputs. During our experiments, as a starting point we stopped the underlying original CNN at 0.72 accuracy and compared numerous different settings.
