import numpy as np

class LinearLayer:
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
        

class ConvLayer:
       def __init__(self, k_dim = 3, stride = 1, padding = 1):
        self.k_dim = k_dim
        self.stride = stride
        self.filter = np.random.uniform(-0.5, 0.5, (self.k_dim, self.k_dim))
        self.filter = self.filter/self.filter.sum() #normalizing the filter
        self.bias = np.zeros((1, 1))
        self.padding = padding

class NN:
    def __init__(self, layers, function='relu'):
        self.structure = layers
        self.input_size = layers[0]
        self.layers = [LinearLayer(n_input=layers[i], n_output=layers[i + 1]) for i in range(len(layers)-2)]
        self.layers.append(LinearLayer(n_input=layers[-2], n_output=layers[-1], activation='softmax'))
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
                    # Now you have the gradients, update the parameters
                    layer.W -= lr * dW
                    layer.b -= lr * db

            # print accuracy for epoch
            print(f'Epoch {epoch} accuracy: {n_correct / len(Y) * 100: .3f}%')

    def test(self,x_input):
        x = x_input
        for layer in self.layers:
            x = layer.forward(x)
        
        return x

    
class CNN:
    def __init__(self) -> None:
        pass

    def train(self):
        pass

    def test(self):
        pass
