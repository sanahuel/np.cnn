import numpy as np

class FCLayer:
    def __init__(self, n_input, n_output, activation='relu'):
        self.W = np.random.randn(n_output, n_input) * np.sqrt(2 / (n_output + n_input)) # Xavier/Glorot initialization for weights
        self.b = np.zeros((n_output, 1))
        self.activation = activation
        self.cache = None
        self.cache_act = None

    def __repr__(self):
        return f'W: {self.W.shape}, b: {self.b.shape} with activation: {self.activation}'

    def relu(self, x, derivative = False):
        if derivative: return np.where(x>0, 1, 0)
        return np.maximum(0, x)

    def sigmoid(self, x, derivative = False):
        np.clip(x, -500, 500)
        if derivative: return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1 / (1 + np.exp(-x))

    def softmax(self,x, derivative = False):
        exps = np.exp(x - x.max())
        if derivative: return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def forward(self, x):
        self.cache = x
        self.cache_act = self.W @ x + self.b
        if self.activation == 'relu':
            x = self.relu(self.W @ x + self.b)
        elif self.activation == 'sigmoid':
            x = self.sigmoid(self.W @ x + self.b)
        elif self.activation == 'softmax':
            x = self.softmax(self.W @ x + self.b)

        return x

    def backward(self, delta):
        if self.activation == 'relu':
            delta = delta * self.relu(self.cache_act, derivative=True)
        elif self.activation == 'sigmoid':
            delta = delta * self.sigmoid(self.cache_act, derivative=True)
        elif self.activation == 'softmax':
            delta = self.softmax(self.cache_act, derivative=True) * delta

        dW = delta @ self.cache.T
        db = np.sum(delta, axis=1, keepdims=True)
        dx = self.W.T @ delta

        return dx, dW, db
        
class Flatten:
    def __init__(self):
        self.input_shape = None

    def __repr__(self):
        return 'Flatten Layer'

    def forward(self, x):
        self.input_shape = x.shape
        return x.flatten().reshape(-1, 1)

    def backward(self, delta):
        return delta.reshape(self.input_shape)

class ConvLayer:
    def __init__(self, k_dim = 3, stride = 1, padding = 1):
        self.k_dim = k_dim
        self.stride = stride
        self.filter = np.random.randn(self.k_dim, self.k_dim) * np.sqrt(2 / (self.k_dim * self.k_dim))
        # self.filter = self.filter/self.filter.sum() #normalizing the filter
        self.bias = np.random.randn(1, 1) * 0.01
        self.padding = padding
        self.cache = None
        self.cache_act = None

    def __repr__(self):
        return f'k_dim: {self.k_dim}, stride: {self.stride}, padding: {self.padding}'
    
    def forward(self, x):
        # padding
        if self.padding > 0:
            pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding))
            x = np.pad(x, pad_width, 'constant', constant_values=0)
        
        self.cache = x

        # input dimensions
        channels, height, width = x.shape

        # output dimensions
        out_height = (height - self.k_dim) // self.stride + 1
        out_width = (width - self.k_dim) // self.stride + 1

        # initialize output
        output = np.zeros((1, out_height, out_width))

        for i in range(0, height - self.k_dim + 1, self.stride):
            for j in range(0, width - self.k_dim + 1, self.stride):
                receptive_field = x[:, i:i + self.k_dim, j:j + self.k_dim]
                output[0, i // self.stride, j // self.stride] = np.sum(receptive_field * self.filter) + self.bias

        # Apply ReLU activation function
        output = np.maximum(0, output)

        self.cache_act = output

        return output

    def backward(self, delta):
        # Compute the gradient with respect to the filter (dW) and bias (db)

        # Apply the gradient of the ReLU activation function
        delta = delta * (self.cache_act > 0)

        # Gradients with respect to the filter (convolution operation)
        dW = np.zeros_like(self.filter)
        for i in range(0, self.cache.shape[1] - self.k_dim + 1, self.stride):
            for j in range(0, self.cache.shape[2] - self.k_dim + 1, self.stride):
                receptive_field = self.cache[:, i:i + self.k_dim, j:j + self.k_dim]
                dW += np.sum(receptive_field * delta[0, i // self.stride, j // self.stride])

        # Gradient with respect to the bias
        db = np.sum(delta)

        # Backpropagate the gradient to the previous layer
        dx = np.zeros_like(self.cache)
        for i in range(0, self.cache.shape[1] - self.k_dim + 1, self.stride):
            for j in range(0, self.cache.shape[2] - self.k_dim + 1, self.stride):
                dx[:, i:i + self.k_dim, j:j + self.k_dim] += self.filter * delta[0, i // self.stride, j // self.stride]

        return dx, dW, db

        
    

class NN:
    def __init__(self, layers, function='relu'):
        self.structure = layers
        self.input_size = layers[0]
        self.layers = [FCLayer(n_input=layers[i], n_output=layers[i + 1]) for i in range(len(layers)-2)]
        self.layers.append(FCLayer(n_input=layers[-2], n_output=layers[-1], activation='softmax'))
        self.function = function.lower()

    def __repr__(self):
        for layer in self.layers:
            print(layer.__repr__())
        return f'NN structure: {self.structure}'

    def train(self, X, Y, epochs, lr=0.01):
        print(f'Training for {epochs} epochs...')
        for epoch in range(epochs):
            n_correct = 0
            for x_input, y in zip(X, Y):
                # forward
                x = x_input
                for layer in self.layers:
                    x = layer.forward(x)

                # count for accuracy
                n_correct += int(np.argmax(x) == np.argmax(y))

                # backpropagation
                loss_gradient = x - y
                for layer in reversed(self.layers):
                    loss_gradient, dW, db = layer.backward(loss_gradient)
                    # update the parameters
                    layer.W -= lr * dW
                    layer.b -= lr * db

            # print accuracy for epoch
            print(f'Epoch {epoch} accuracy: {n_correct / len(Y) * 100: .3f}%')

    def test(self, X, Y):
        n_correct = 0
        for x, y in zip(X, Y):
            for layer in self.layers:
                x = layer.forward(x)
            
            # count for accuracy
            n_correct += int(np.argmax(x) == np.argmax(y))

        print(f'Test accuracy: {n_correct / len(Y) * 100: .3f}%')

    
class CNN:
    def __init__(self, layers) -> None:
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, X, Y, epochs, lr=0.01):
        print(f'Training for {epochs} epochs...')
        for epoch in range(epochs):
            n_correct = 0
            for x_input, y in zip(X, Y):
                # forward
                x = x_input
                for layer in self.layers:
                    x = layer.forward(x)

                # count for accuracy
                n_correct += int(np.argmax(x) == np.argmax(y))

                # backpropagation
                loss_gradient = x - y
                for layer in reversed(self.layers):
                    if isinstance(layer, FCLayer):
                        loss_gradient, dW, db = layer.backward(loss_gradient)
                        layer.W -= lr * dW
                        layer.b -= lr * db
                    elif isinstance(layer, ConvLayer): 
                        loss_gradient, dW, db = layer.backward(loss_gradient)
                        layer.filter -= lr * dW
                        layer.bias -= lr * db
                    elif isinstance(layer, Flatten): 
                        loss_gradient = layer.backward(loss_gradient)
                
                # print(loss_gradient)

            # print accuracy for epoch
            print(f'Epoch {epoch} accuracy: {n_correct / len(Y) * 100: .3f}%')


    def test(self, X, Y):
        n_correct = 0
        for x, y in zip(X, Y):
            for layer in self.layers:
                x = layer.forward(x)
            
            # count for accuracy
            n_correct += int(np.argmax(x) == np.argmax(y))

        print(f'Test accuracy: {n_correct / len(Y) * 100: .3f}%')
