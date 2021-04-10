# Fashion_MNIST_TES-21

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training
set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image,
associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in
replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the
same image size and structure of training and testing splits.

Brief description of deep learning model

Deep learning is an increasingly popular subset of machine learning. Deep learning models are built using neural networks. A neural network takes in inputs, which are then processed in hidden layers using weights that are adjusted during training. Then the model spits out a prediction. The weights are adjusted to find patterns in order to make better predictions. 

About the project
In my project, I have used Google colab with GPU (Graphical Processing Unit) for computations and use ReLU (Rectified Linear Activation) and Softmax activation functions. The dataset was downloaded from the drive and was uploaded on to my drive which was then mounted to Google colab.

Training the neural network and Tests


The neural network is trained for backward propagation to optimize weights and bias.
The cost or loss function has an important job in that it must faithfully distill all aspects of the model down into a single number in such a way that improvements in that number are a sign of a better model. The model was trained for 5 epochs and a learning rate of 0.001, the loss was about 0.2983.

The test accuracy was 87.47%
