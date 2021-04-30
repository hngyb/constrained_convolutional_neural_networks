# constrained_convolutional_neural_networks
This is an unofficial repository implementing "Constrained Convolutional Neural Networks: A New Approach Towards General Purpose Image Manipulation Detection".
## MISLnet
![MISLnet](https://user-images.githubusercontent.com/68726615/116695349-72a16f80-a9fb-11eb-948b-ea7ec80cd166.png)
1. **Prediction Error Feature Extraction**
    - Constrained convolutional layer suppresses the content and constrains CNN to learn prediction error features in the first layer.
    - Patch sized 256 X 256 from a grayscale input image is first convolved with three different 5 X 5 constrained convolutional filters with a stride equal to 1.
    - Yields feature maps of prediction residuals of dimension 252 X 252 X 3.
2. **Hierarchical Feature Extraction**
    - Three consecutive convolutional layers each followed by a batch normalization, activation function and pooling layers.
    - Conv2: 96 filters of size 7 X 7 X 3 and stride of 2
    - Conv3: 64 filters of size 5 X 5 X 96 and stride of 1
    - Conv4: 64 filters of size 5 X 5 X 64 and stride of 1.
    - Activation function: TanH
    - Pooling: Three max-pooling and one average-pooling
3. **Cross Feature Maps Learning**
    - In order to constrain CNN to learn only association across feature maps, we use 1 X 1 convolutional layer after the hierarchical feature extraction conceptual block.
    - Conv5: 128 different 1 X 1 convolutional filters with stride of 1
    - The output dimension of this convolutional layer is 15 X 15 X 128.
4. **Classification**
    - One neuron for each possible tampering operation and another neuron that corresponds to the unaltered image class.
