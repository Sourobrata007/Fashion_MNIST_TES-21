# Project Level-1: Fashion_MNIST_TES-21

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


# Project Level-2: FOOD_MNIST_TES_21
Food-MNIST is a dataset with 10 food categories, with 5,000 images. For each
class, 125 manually reviewed test images are provided as well as 375 training images. All images were
rescaled to have a maximum side length of 512 pixels.
The dataset you are getting is a subset of the above mentioned dataset and has only 4 classes.
● 0 : apple pie
● 1 : baby back ribs
● 2 : baklava
● 3 : beef carpaccio

# *Model_1* 

The first model is Convolutional Neural Network based on Support Vector Machine. It has a Convolutional layer, Pooling layer, second, third convolutional layer, flattening layer, second dense layer and finally an output layer. All the six layers have Relu activation function and the final, output layer has adam optimiser and squared hinge loss function. Summary : Total params: 103,896
Trainable params: 103,896
Non-trainable params: 0
I ran 100 epochs on the model and the graphs for loss and accuracy were plotted.


# *Model_2*

The second model is based on Inception pretrained model, which has efficient architectures such as ResNet, etc. Model Summary :Total params: 22,065,572
Trainable params: 22,031,140
Non-trainable params: 34,432
The model was run for 30 epochs and gave 96.5% test accuracy score.
