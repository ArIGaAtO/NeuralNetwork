"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import matplotlib.pyplot as plt

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""              
        self.num_layers = len(sizes)
        self.sizes = sizes
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.highest = 0

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        a = a.T#np.matrix()
        #print(a.shape)
        #print("a0")
        i = 1
        #print(len(self.biases))
        #print(self.num_layers)
        for b, w in zip(self.biases, self.weights):
            #a = linear_activation(np.dot(w, a)+b)
            #a = sigmoid(np.dot(w, a)+b)
            if i<(self.num_layers-1):
                a = relu(np.dot(w, a)+b)
                #a = leaky_relu(np.dot(w, a)+b)
                #a = linear_activation(np.dot(w, a)+b)
            else:
                a = sigmoid(np.dot(w, a)+b)
                #a = linear_activation(np.dot(w, a)+b)
            #a = relu(np.dot(w, a)+b)
            i+=1
        return np.array(a.T)[0]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if not(test_data is None): 
            n_test = len(test_data)
        n = len(training_data)
        training_errors = []
        fakeTraining_errors = []
        fakeTraining_inRange = []
        epochsForGraph = []
        for j in range(epochs):#xrange
            random.shuffle(training_data)#Probably very ineffective
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]#xrange
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if not(test_data is None):
                realProgressData = self.evaluate(test_data)
                fakeProgressData = self.evaluate(training_data[:30])
                #print(fakeProgressData[0])
                #print(fakeProgressData[2]/30)
                print ("Epoch {0}: {1} / {2}        {3}".format(j+1, round(realProgressData[0],1), n_test*5, round(realProgressData[1]/150,4)))                
                epochsForGraph.append(j)
                training_errors.append(realProgressData[1]/150)    
                fakeTraining_errors.append(fakeProgressData[1]/150)  
                fakeTraining_inRange.append(fakeProgressData[0])         
                if round(self.evaluate(test_data)[0],1) > self.highest:
                    self.highest = round(self.evaluate(test_data)[0],1)
            else:
                print ("Epoch {0} complete".format(j))

        plot_training_progress(epochsForGraph, training_errors, fakeTraining_errors, fakeTraining_inRange)
        #plot_training_progress(epochsForGraph, fakeTraining_errors)
        print(self.highest)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]"""
        #test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        test_results = []
        for (x, y) in test_data:
            test_results.append((self.feedforward(x), y))               
        num = 0
        howOFOnAvarage = 0
        howOFOnAvarage1 = 0
        for (x, y) in test_results:
            #howOFOnAvarage = howOFOnAvarage + abs(sum(x) - sum(y))**2
            #howOFOnAvarage1 = howOFOnAvarage1 + abs(sum(x) - sum(y))                  
            for i in range(len(x)): 
                ##print(abs(x[i] - y[i]))             
                howOFOnAvarage1 = howOFOnAvarage1 + abs(x[i] - y[i])
                if abs(x[i] - y[i]) < 0.1:
                    num = num + 1
        #print(howOFOnAvarage1)
        #print(len(test_results))
        return [num,howOFOnAvarage1] #sum(int(x == y) for (x, y) in test_results)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: #X are inputs    and     Y are correct outputs
            delta_nabla_b, delta_nabla_w = self.backprop(x.T, y.T)#np.matrix(x).T, np.matrix(y).T
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = np.array(x)[np.newaxis].T
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        i = 1
        for b, w in zip(self.biases, self.weights):           
            z = np.dot(w, activation) + b#np.matrix(activation)
            zs.append(z)
            #activation = linear_activation(z)
            #activation = sigmoid(z)
            #activation = leaky_relu(z)
            if i<(self.num_layers-1):
                activation = relu(z)
                #activation = leaky_relu(z)
                #activation = linear_activation(z)
            else:
                activation = sigmoid(z)
                #activation = linear_activation(z)
            #a = relu(np.dot(w, a)+b)
            i+=1
            #activation = relu(z)
            activations.append(activation)
        # backward pass
        #print()
        #delta = np.array([k*l for k, l in zip(self.cost_derivative(activations[-1], y[np.newaxis].T), linear_activation_prime(zs[-1]))]).T
        #sigmoid_prime(z)
        
        delta = np.array([k*l for k, l in zip(self.cost_derivative(activations[-1], y[np.newaxis].T), sigmoid_prime(zs[-1]))]).T#np.matrix()
        #delta = np.array([k*l for k, l in zip(self.cost_derivative(activations[-1], y[np.newaxis].T), linear_activation_prime(zs[-1]))]).T
        #print(sigmoid_prime(zs[-1]))
        #delta0 = np.array([k*l for k, l in zip(self.cost_derivative(activations[-1], y[np.newaxis].T), leaky_relu_prime(zs[-1]))]).T
        #print(leaky_relu_prime(zs[-1]))
        
        #delta = (self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])).T#np.matrix()
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)#.transpose()
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):#xrange
            z = zs[-l]
            #sp = linear_activation_prime(z)
            #sigmoid_prime(z)
            #sp = sigmoid_prime(z)
            #sp = leaky_relu_prime(z)
            sp = relu_prime(z)
            delta = np.array([k*l for k, l in zip(np.dot(self.weights[-l+1].T, delta), sp)]).T#np.matrix()           
            #nabla_b[-l] = delta
            nabla_b[-l] = clip_gradients(delta,1000)
            #nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_w[-l] = clip_gradients(np.dot(delta, activations[-l-1].transpose()), 100)
            

        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""      
        
        return (output_activations-y)

def clip_gradients(grad, threshold):

    return np.clip(grad, -threshold, threshold)
    #return np.clip(grad, 0, threshold)

def plot_training_progress(epochs, training_errors, fakeTraining_errors, fakeTraining_inRange):
    plt.figure(figsize=(10, 6))
    #print(fakeTraining_inRange)
    plt.plot(epochs, training_errors, label='Úspěšnost na testovacích vstupech', color='blue')
    plt.plot(epochs, fakeTraining_errors, label='Úspěšnost na tréninkovích vstupech', color='red')
    plt.title('Trénovací progres')
    plt.xlabel('Epocha')
    plt.ylabel('Průměrná chyba')
    plt.legend()
    plt.grid(True)
    plt.show()
    

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    myArray = []
    for i in z:    
        myArray.append(sigmoid(i)*(1-sigmoid(i)))
    return np.array(myArray).T#sigmoid(z)*(1-sigmoid(z))#np.matrix()

def leaky_relu(z):
    myArray = []
    #print("z in leaky_relu")
    #print(z.shape) 
    if z.shape[1] == 1:
        for i in z:
            if i>0 :
                myArray.append(i)
            else :
                #myArray.append(0)
                myArray.append(0.01*i)
    else:
        for i in z:
            myArray0 =[]
            for j in i:
                if j>0 :
                    myArray0.append(j)
                else :
                    myArray0.append(0.01*j)
                    #myArray0.append(0)  
            myArray.append(myArray0)   
    return np.array(myArray)#, dtype=float)
def leaky_relu_prime(z):
    myArray = []
    for i in z:
        if i>=0 :
            myArray.append([1])
        else :
            myArray.append([0.01])
    #print(np.array(myArray))
    #return np.array(myArray, dtype=float).T
    
    return np.array(myArray).T

def relu(z):
    myArray = []
    #print("z in leaky_relu")
    #print(z.shape) 
    if z.shape[1] == 1:
        for i in z:
            if i>0 :
                myArray.append(i)
            else :
                #myArray.append(0)
                myArray.append(0*i)
    else:
        for i in z:
            myArray0 =[]
            for j in i:
                if j>0 :
                    myArray0.append(j)
                else :
                    myArray0.append(0*j)
                    #myArray0.append(0)  
            myArray.append(myArray0)   
    return np.array(myArray)#, dtype=float)
def relu_prime(z):
    myArray = []
    for i in z:
        if i>=0 :
            myArray.append([1])
        else :
            myArray.append([0])
    #print(np.array(myArray))
    #return np.array(myArray, dtype=float).T
    return np.array(myArray).T

def linear_activation(z):
    return z
def linear_activation_prime(z):
    #myArray = np.zeros(z.shape)
    myArray = []
    for i in z:
        myArray.append([1])
    #print("linear")
    #print(myArray)
    return np.array(myArray).T#[np.newaxis]


def get_training_data5(how_Much_Data):
    theTrainingSet = []
    for i in range(how_Much_Data):
        theInputAndOutput = []
        theInput = []
        theOutput = []        
        for j in range(5):
            #print(random.random())
            theOutput.append(random.random())
        
        theOutput.sort()
        
        theInput.append(theOutput[0]*theOutput[1]*theOutput[2]*theOutput[3]*theOutput[4])
        theInput.append(- theOutput[0] - theOutput[1] - theOutput[2] - theOutput[3] - theOutput[4])
        theInput.append(theOutput[0]*theOutput[1] + theOutput[0]*theOutput[2] + theOutput[0]*theOutput[3] + theOutput[0]*theOutput[4]  + theOutput[1]*theOutput[2] + theOutput[1]*theOutput[3] + theOutput[1]*theOutput[4] + theOutput[2]*theOutput[3] + theOutput[2]*theOutput[4] + theOutput[3]*theOutput[4])
        theInput.append(- theOutput[0]*theOutput[1]*theOutput[2] - theOutput[0]*theOutput[1]*theOutput[3] - theOutput[0]*theOutput[1]*theOutput[4] - theOutput[0]*theOutput[2]*theOutput[3] - theOutput[0]*theOutput[2]*theOutput[4] - theOutput[0]*theOutput[3]*theOutput[4] - theOutput[1]*theOutput[2]*theOutput[3]  - theOutput[1]*theOutput[2]*theOutput[4] - theOutput[1]*theOutput[3]*theOutput[4] - theOutput[2]*theOutput[3]*theOutput[4])
        theInput.append(theOutput[0]*theOutput[1]*theOutput[2]*theOutput[3] + theOutput[0]*theOutput[1]*theOutput[2]*theOutput[4] + theOutput[0]*theOutput[1]*theOutput[3]*theOutput[4] + theOutput[0]*theOutput[2]*theOutput[3]*theOutput[4] + theOutput[1]*theOutput[2]*theOutput[3]*theOutput[4])
        
        
        theInputAndOutput.append(theInput)
        theInputAndOutput.append(theOutput)
        

        theTrainingSet.append(theInputAndOutput)
    return theTrainingSet
def get_training_data2(how_Much_Data):
    theTrainingSet = []
    for i in range(how_Much_Data):
        theInputAndOutput = []
        theInput = []
        theOutput = []        
        for j in range(2):
            #print(random.random())
            theOutput.append(random.random())
        theInput.append(theOutput[0]*theOutput[1])
        theInput.append(- theOutput[0] - theOutput[1])
        
        
        
        theInputAndOutput.append(theInput)
        theInputAndOutput.append(theOutput)
        

        theTrainingSet.append(theInputAndOutput)
    return theTrainingSet
def get_training_data1(how_Much_Data):
    theTrainingSet = []
    for i in range(how_Much_Data):
        theInputAndOutput = []
        theInput = []
        theOutput = []             
        theOutput.append(random.random())

        theInput.append(theOutput[0])
                             
        theInputAndOutput.append(theInput)
        theInputAndOutput.append(theOutput)
        

        theTrainingSet.append(theInputAndOutput)
    return theTrainingSet

theTrainingSet = get_training_data5(1030)


training_data = []
test_data = []
for i in range(len(theTrainingSet)):
    if i < (len(theTrainingSet) - 30):
        training_data.append(theTrainingSet[i])
    else:
        test_data.append(theTrainingSet[i])

training_data = np.array(training_data)
test_data = np.array(test_data)

net1 = Network([5, 16, 16, 5])
net1.SGD(training_data, 100, 10, 0.002, test_data=test_data)

for i in range(100):
    theTestingSet = get_training_data5(10)
    #for j in range(len(theTestingSet)):
        #print(j,": ",theTestingSet[j])
    selected = theTestingSet[int(input())]
    print()
    print()
    print(selected[0])
    print(selected[1])
    print(net1.feedforward(np.array(selected[0])))


