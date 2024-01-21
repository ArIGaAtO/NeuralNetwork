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
        
        self.best_testing_biases = self.biases
        self.best_testing_weights = self.weights
        self.testing_highest = 0
        

        self.best_training_biases = self.biases
        self.best_training_weights = self.weights
        self.training_highest = 0
        
        self.lowest_testing_biases = self.biases
        self.lowest_testing_weights = self.weights
        self.testing_lowest_error = 1000
        
        self.lowest_training_biases = self.biases
        self.lowest_training_weights = self.weights
        self.training_lowest_error = 1000


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
                #a = sigmoid(np.dot(w, a)+b)#not last layer
                #a = relu(np.dot(w, a)+b)
                a = leaky_relu(np.dot(w, a)+b)
                #a = linear_activation(np.dot(w, a)+b)
            else:
                a = sigmoid(np.dot(w, a)+b)
                #a = linear_activation(np.dot(w, a)+b)
            #a = relu(np.dot(w, a)+b)
            i+=1
        return np.array(a.T)[0]

    def SGD(self, list_training_data, epochs, mini_batch_size, eta, test_data=None):
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
        n = len(list_training_data)
        training_errors = []
        fakeTraining_errors = []
        fakeTraining_inRange = []
        epochsForGraph = []
        
        for j in range(epochs):#xrange
            """print("před")
            for i in list_training_data:
                print(i[0])"""
            random.shuffle(list_training_data)#Probably very ineffective
            training_data = np.array(list_training_data)
            """print("po")
            for i in training_data:
                print(i[0])
            break"""
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]#xrange
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if not(test_data is None):
                realProgressData = self.evaluate(test_data)
                fakeProgressData = self.evaluate(training_data[:30])
                #print(fakeProgressData[0])
                #print(fakeProgressData[2]/30)
                print ("Epoch {0}: {1} / {2}        {3}         {4} / {5}        {6}".format(j+1, realProgressData[0], n_test*5, round(realProgressData[1]/150,4), fakeProgressData[0], n_test*5, round(fakeProgressData[1]/150,4)))                
                epochsForGraph.append(j)
                training_errors.append(realProgressData[1]/150)    
                fakeTraining_errors.append(fakeProgressData[1]/150)  
                fakeTraining_inRange.append(fakeProgressData[0])         
                if realProgressData[0] > self.testing_highest:
                    self.testing_highest = realProgressData[0]
                    self.best_testing_biases = self.biases
                    self.best_testing_weights = self.weights
                if fakeProgressData[0] > self.training_highest:
                    self.training_highest = fakeProgressData[0]
                    self.best_training_biases = self.biases
                    self.best_training_weights = self.weights
                
                if realProgressData[1]/150 > self.testing_lowest_error:
                    self.testing_lowest_error = realProgressData[1]/150
                    self.lowest_testing_biases = self.biases
                    self.lowest_testing_weights = self.weights
                if fakeProgressData[1]/150 > self.training_lowest_error:
                    self.training_lowest_error = fakeProgressData[1]/150
                    self.lowest_training_biases = self.biases
                    self.lowest_training_weights = self.weights
            else:
                print ("Epoch {0} complete".format(j))

        plot_training_progress(epochsForGraph, training_errors, fakeTraining_errors, fakeTraining_inRange)
        #plot_training_progress(epochsForGraph, fakeTraining_errors)
        print(self.testing_highest)
        print(self.training_highest)

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
                #activation = sigmoid(z)#not last layer
                #activation = relu(z)
                activation = leaky_relu(z)
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
            #sp = sigmoid_prime(z)#not last layer
            sp = leaky_relu_prime(z)
            #sp = relu_prime(z)
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
    plt.plot(epochs, fakeTraining_errors, label='Úspěšnost na tréninkových vstupech', color='red')
    plt.title('Učební progres sítě')
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


list_training_data = []
test_data = []
for i in range(len(theTrainingSet)):
    if i < (len(theTrainingSet) - 30):
        list_training_data.append(theTrainingSet[i])
    else:
        test_data.append(theTrainingSet[i])

#training_data = np.array(training_data)
test_data = np.array(test_data)

new_biasesF1 = [np.array([[-0.38975191],
       [-0.11167123],
       [ 0.02465081],
       [ 1.49254061],
       [-0.62483725],
       [-1.91130523],
       [-0.77955054],
       [-0.18410759],
       [ 1.4693215 ],
       [ 0.13406283],
       [ 0.35143403],
       [-0.46076108],
       [-0.10191113],
       [-0.10369742],
       [-1.86827607],
       [ 1.85190239]]), np.array([[-0.31990021],
       [-0.2308423 ],
       [-0.35530156],
       [-0.12054634],
       [ 0.99498225],
       [ 1.50051237],
       [-0.62843137],
       [-0.82162845],
       [ 0.61799106],
       [ 0.29330975],
       [-0.57976284],
       [-0.23643368],
       [-1.37768655],
       [-1.08523392],
       [ 0.21404974],
       [-1.22090818]]), np.array([[-0.5981304 ],
       [ 1.05825159],
       [-0.72121293],
       [ 3.47949634],
       [-0.54081928],
       [ 0.77036265],
       [ 0.59735111],
       [-0.83043287],
       [ 1.17183582],
       [ 2.19595753],
       [-0.87817941],
       [-1.86230916],
       [ 1.58316792],
       [-0.85495062],
       [ 2.03082702],
       [ 1.3528998 ]]), np.array([[-1.38518861],
       [-0.96799083],
       [ 0.91120141],
       [ 0.50982897],
       [ 1.353478  ]])]
new_weightsF1 = [np.array([[ 0.16739261,  0.09993713,  0.48437892,  0.02456783,  0.41942293],
       [-0.12838826, -0.74230698,  1.03006034,  0.23575451,  0.4389864 ],
       [ 0.82258775, -0.14749545, -0.10310791, -0.85808817,  1.17023772],
       [ 0.08745645,  0.43859869, -0.33157017,  0.27716748,  0.66669493],
       [-0.15957943, -0.13060244, -0.79865326,  0.01927549, -1.01993271],
       [-0.81188836, -0.47975775,  0.27829351, -0.98944954,  0.4812733 ],
       [-0.18032791,  0.13647075,  0.33232586, -0.41506038,  0.13550781],
       [ 0.37147468, -0.07521173, -0.25309603,  0.77427588,  0.57757247],
       [ 0.83821584,  0.464899  , -0.01678674,  0.24590121, -0.16820311],
       [-0.80155804,  0.52000198,  1.05864655,  0.86422056,  0.4916672 ],
       [ 0.25677075,  0.48637033,  1.09308393,  0.41291221,  1.35150387],
       [-0.11023039,  0.58556196,  0.00854475,  1.43395274, -0.32760443],
       [ 1.3390824 ,  1.12107193, -0.29712553,  0.71769606,  0.94161483],
       [-0.45268461,  0.02810282, -1.04198387, -0.47309499,  0.53529247],
       [-0.50202464,  0.95130904,  2.03943288, -0.62578492, -0.15737965],
       [ 0.12251991, -0.04743867, -0.13887657, -0.56418682, -0.66993724]]), np.array([[-0.26192622, -0.42501288, -0.16376704, -0.42176146, -0.40503993,
        -0.20386931,  0.57452379, -0.18902432, -0.45765159,  0.0847119 ,
         0.26593103,  0.04601032,  0.13125135,  0.04943416,  0.014285  ,
        -0.14202517],
       [-0.3170775 ,  0.14830119,  0.43184312,  0.22502453, -0.4232821 ,
        -0.43955145, -0.2412814 ,  0.28149945,  0.11375636, -0.21401055,
        -0.68913849,  0.33242922,  0.21469739, -0.32241532, -0.46958748,
        -0.21959682],
       [-0.01856796, -0.37970826,  0.11122261, -0.04030753,  0.05852363,
        -0.23679476, -0.00599246, -0.21014482, -0.41825623,  0.03735715,
        -0.36253898,  0.21365952, -0.81028165, -0.02628436, -0.02976898,
         0.56756783],
       [ 0.3489732 ,  0.88885053,  0.68124591,  0.09165929,  0.01596156,
         1.00549377,  0.14086917,  0.21813828,  0.16090929, -0.05400607,
         0.58220942, -0.27013833,  0.0981696 ,  0.4708486 ,  1.40580002,
        -0.09338407],
       [-0.05332496, -0.15162046, -0.95718309, -0.17628411,  0.04689406,
         0.09036022, -0.03910936,  0.40022405, -0.24237273, -0.55827007,
         0.28343175,  0.2503955 ,  0.17220371, -0.19803294, -0.45520381,
        -0.32260553],
       [ 0.19840452, -0.36598428,  0.19951994,  0.17398178,  0.12941768,
        -0.38596745,  0.34706885,  0.13099641, -0.2656084 , -0.38163443,
         0.29884178, -0.32931045, -0.12761601,  0.17659879, -0.26230261,
        -0.73338436],
       [ 0.28411581, -0.28035099, -0.12246032,  0.48350499, -0.3387643 ,
         0.39188532, -0.19821227, -0.84443886, -0.1131258 ,  0.43842631,
        -0.07438525,  0.73890917,  0.50011687, -0.08711972, -0.59295316,
         0.28825525],
       [ 0.19657207, -0.43873455,  0.03716359, -0.20125233,  0.28975298,
        -0.67892031, -0.34784598, -0.40776558,  0.1562409 , -0.43374066,
        -0.6999618 ,  0.31236812,  0.30805636, -0.13741817, -0.07926936,
         0.22669237],
       [-0.02893879, -0.59426337,  0.01265742,  0.25049347, -0.11634459,
        -0.82640499,  0.19793453,  0.31058964, -0.31512114,  0.69472166,
         0.33322497,  1.12209697, -0.21522793,  0.68708078,  0.13714822,
        -0.0712476 ],
       [-0.57265447, -0.50235517,  0.26718578, -0.374854  , -0.26718913,
         0.16049704, -0.7650715 , -0.66742537, -0.26771547, -0.39003813,
        -0.1470757 ,  0.31526954,  0.32769258,  0.2416748 , -0.13699063,
         0.29632923],
       [-0.06219105, -0.19273474, -0.75845857, -0.16119618,  0.21908713,
        -0.55223755, -0.28681037, -0.28340399,  0.41781402,  0.9998916 ,
        -0.08865262,  0.1770713 , -0.13201793, -0.30046404,  0.33273937,
        -0.47764775],
       [-0.56474859, -0.81156859,  0.06736012,  0.25027858, -0.70970514,
         0.37668536, -0.21446886, -0.06826696, -0.32487877,  0.17655043,
        -0.25445156,  0.17679886, -0.04269094, -0.45722598, -0.23612212,
         0.40937186],
       [ 0.1940127 , -0.35717702, -0.29382237,  0.51102683, -0.41203538,
        -0.62703705, -0.67103388, -0.141294  ,  0.48488142,  0.32040242,
         0.58894502, -0.02100672,  0.05877873,  0.02953424,  0.3159453 ,
        -0.11679519],
       [ 0.22980062,  0.10841401, -0.07738321, -0.44951768, -0.78800146,
        -0.19549134,  0.2571209 , -0.48519258, -0.21755857, -0.10168902,
         0.14645667, -0.06419108, -0.11570397, -0.20454446, -0.25972182,
        -0.21487019],
       [ 0.128788  , -1.11680004,  0.34304033,  0.48028099, -0.85444221,
         0.16600069, -0.2646502 ,  0.31441777, -0.38678944, -0.01425171,
        -0.62821138,  0.5719551 , -0.39945149, -0.5220029 ,  0.57367921,
         0.00890292],
       [-0.21937149, -1.05621335, -0.07327411,  0.03194313, -0.42486923,
         0.22259685,  0.05516805, -0.24553269,  0.04310846,  0.55185853,
         0.46197382,  0.2745509 ,  0.03662241,  0.5205327 , -0.42726921,
         0.0028357 ]]), np.array([[ 8.44575007e-02, -2.74039370e-01, -3.78857427e-01,
        -7.05149258e-01, -7.35006684e-02, -1.81202888e-01,
        -4.50175861e-01, -8.48899334e-02,  1.75763737e-01,
         1.80930609e-01,  1.56409822e-01, -4.76259294e-02,
        -6.56155215e-02,  5.72738804e-01,  3.88469646e-01,
        -4.10880863e-02],
       [-8.46900909e-01,  5.86018014e-01,  5.94353679e-03,
        -7.98681340e-01, -3.45627858e-02, -5.71967361e-02,
        -1.32493624e-01, -8.56132916e-02, -7.76027712e-02,
         2.88405819e-01,  2.30710061e-01, -3.20624672e-02,
         2.28150279e-01,  6.77731084e-01, -2.72660120e-01,
        -1.58915637e-01],
       [-9.17753000e-01, -3.07364182e-01,  3.65995190e-01,
        -5.93431549e-01, -3.87257507e-01,  8.78433330e-03,
        -2.41568066e-01, -2.79748073e-01,  5.83586835e-01,
        -2.43683593e-01, -6.73951753e-02,  1.61376362e-01,
        -6.43057657e-02, -5.16192336e-01, -7.20531090e-01,
         1.91172808e-01],
       [-4.26029911e-01, -2.57222772e-01,  3.05236032e-01,
        -3.25131447e-01,  6.49518696e-01,  1.96022880e-01,
         1.36751081e-01,  1.87401614e-01, -6.96735490e-02,
        -1.22963434e-01, -8.32463221e-01, -2.34803137e-01,
        -3.95824953e-01,  2.62925049e-01, -2.60797317e-01,
        -3.45463852e-01],
       [-5.09779127e-02,  1.38434373e-01, -4.64978551e-01,
        -3.22232884e-01, -5.18290303e-02, -1.87270790e-01,
        -2.00195617e-01, -9.54649057e-02,  6.62313312e-01,
        -3.69782919e-01,  2.55479991e-01,  6.07518571e-01,
         2.05845262e-01,  8.90793713e-02,  2.17333118e-01,
         1.52657279e-01],
       [-7.60158341e-01, -5.87769733e-01, -9.44296056e-04,
        -6.64080137e-01,  1.61385940e-01, -3.69841544e-01,
        -2.62244179e-01, -5.31366552e-02,  1.74281360e-01,
         6.59317984e-01, -2.29793482e-01,  5.75164260e-01,
        -3.14338186e-01,  2.37105773e-01,  4.24530046e-01,
         1.85006835e-01],
       [-7.20867781e-01,  6.15790478e-01,  5.14386066e-02,
        -5.99881206e-01, -3.25051536e-01,  3.69508994e-01,
        -6.28529621e-02,  1.45160316e-01,  6.91519355e-02,
         1.28112904e-01,  1.54252437e-01,  1.86809328e-01,
        -4.09807741e-01, -5.68223400e-01, -3.59632325e-01,
        -1.88988063e-01],
       [ 3.81539588e-01, -1.08058705e-01,  2.98706348e-01,
        -4.38050364e-01,  7.20365753e-01, -5.51110304e-01,
         1.88086390e-01, -1.44456805e-02,  3.09205067e-01,
         8.32235848e-02, -3.63234401e-01, -2.53967107e-01,
        -5.49846150e-01,  6.49350128e-01,  3.08383165e-02,
         3.71173458e-01],
       [-3.92719440e-01, -7.54934322e-02, -6.07880760e-01,
        -8.34176141e-01,  4.17905177e-01,  3.27336966e-02,
        -2.49625647e-01,  9.55815743e-03,  1.74759383e-01,
        -3.14923127e-01,  1.21285990e-01,  6.07194353e-03,
        -2.14183245e-01,  3.97575415e-01, -2.37904075e-01,
         2.65999353e-01],
       [-8.78162353e-02,  2.34296654e-01, -7.46065770e-02,
        -1.07776049e+00,  3.93353722e-01, -1.33690214e-02,
         3.96062579e-01, -3.27525576e-01,  3.44593369e-02,
        -8.19778693e-01,  3.05330010e-01,  4.14392289e-01,
        -1.12095690e-01, -2.78696765e-01, -4.08910030e-01,
         2.58167712e-01],
       [-3.36336410e-01,  2.97320198e-01, -1.66396850e-01,
        -8.05627659e-01,  5.08674963e-01, -4.32181257e-01,
         9.43361419e-03,  3.49208787e-01,  2.79431145e-01,
        -4.86892793e-01, -3.90614695e-01,  1.82835882e-01,
        -1.42430257e-02, -1.80603906e-01, -1.97114050e-01,
         5.91461998e-01],
       [ 3.65602700e-01, -3.28317122e-01,  3.15917690e-01,
        -1.14481954e+00, -4.40217272e-02, -9.64720667e-02,
         3.83587064e-01,  5.75092767e-01,  2.77418450e-02,
         3.98097657e-01,  3.09640615e-01,  2.46890742e-01,
        -4.18675476e-01,  6.37016237e-02, -3.18010950e-01,
        -9.16954402e-01],
       [-1.45440460e-02, -2.66218217e-01, -4.33765383e-01,
        -1.02946117e+00, -5.65232056e-01, -2.35996383e-01,
         9.88401427e-03,  7.26146998e-01,  4.97605054e-01,
         2.99730906e-01, -9.27866683e-02,  2.26767192e-02,
        -5.45511379e-01,  3.45114289e-01, -3.11154254e-01,
         1.10831228e-02],
       [ 5.13177976e-01, -7.11740133e-01, -2.43703071e-01,
        -5.67317019e-01, -3.02852250e-01,  6.91093040e-02,
         2.73238190e-01,  2.93659660e-01, -1.91690650e-01,
        -4.08695998e-01,  7.36739443e-02,  1.34293126e-01,
        -1.61617626e-01, -4.01382486e-01, -4.55298844e-01,
         7.47607767e-01],
       [ 3.59555712e-01,  1.19514983e-01,  4.02371139e-01,
        -6.87091564e-01, -6.37225183e-01, -3.51731529e-02,
        -1.97590058e-01,  5.50182992e-02, -5.26084869e-01,
         7.58928162e-02, -4.03547798e-01,  6.78842488e-02,
         2.67464038e-02, -2.50550787e-01, -6.99689889e-01,
        -3.42346560e-01],
       [-3.17726193e-02, -4.10766801e-02,  7.89126204e-01,
        -9.11194740e-01, -1.93993394e-01,  3.35688623e-01,
        -2.21480830e-01, -2.91680002e-01,  3.53526941e-01,
        -2.59685874e-01, -2.08995147e-01, -2.70026902e-01,
        -3.89985649e-02,  2.08445317e-01,  2.99689916e-02,
         4.44737972e-02]]), np.array([[ 0.33541131, -0.29954759,  0.44002952, -0.31078396, -1.16159618,
         0.17325229, -0.57645978, -0.34925962, -0.28662608, -0.37113149,
        -0.24737994, -0.28158571, -0.36752858, -0.79816383, -0.96869873,
         0.36739592],
       [-1.10617389, -0.56918046,  0.0365625 , -0.07593279, -0.49468953,
        -0.48090039, -0.21829766, -0.47625895,  0.13607835, -0.45002287,
        -1.00927799, -0.36899428, -0.45387127, -0.02328955,  1.15731793,
        -0.37258062],
       [-0.40639118,  0.29970142,  0.25644615,  0.207386  ,  0.14795547,
         0.04211282, -0.69206979, -0.59785285,  0.32548141,  0.16431475,
        -0.6992313 , -0.40185219, -0.32642807, -0.45365409,  0.92843688,
         0.38303925],
       [-0.02077718, -0.03779625,  0.87652107,  0.37839819, -0.12790043,
        -0.26661232, -0.47736541,  0.53381597, -0.15063759,  0.29013229,
        -0.03030002, -0.41841046,  0.3596382 , -0.18472145,  0.94895949,
        -0.30836485],
       [-0.12412095,  0.03638969, -0.10389391,  0.32972504, -0.14875623,
         0.19618867,  0.50808588, -0.02858257,  0.23011226,  0.03280498,
        -0.08232065, -0.44683758,  0.27464704,  0.93816484,  0.94770204,
        -0.79777699]])]

net1 = Network([5, 16,16,16, 5])
net1.biases = new_biasesF1
net1.weights = new_weightsF1
net1.SGD(list_training_data, 100, 20, 0.1, test_data=test_data)

user = input()
while user != "end":
    theTestingSet = get_training_data5(10)
    #for j in range(len(theTestingSet)):
        #print(j,": ",theTestingSet[j])
    testing = True
    try:
        val = int(user)
        if val < 0 or val > 9:
            testing = False
    except ValueError:
        testing = False
    if testing:
        selected = theTestingSet[int(user)]
        print()
        print()
        print(selected[0])
        print(selected[1])
        print(net1.feedforward(np.array(selected[0])))
    elif user == "rwb":
        print("test weights")
        #print(net1.best_testing_weights)  
        print(" [",end="")
        for i in range(len(net1.best_testing_weights)):
            print(" np.array(", end="")
            print(net1.best_testing_weights[i],end="")
            print(")",end="")
            if i < len(net1.best_testing_weights)-1:
                print(",",end="")
        print("]")

        print("test biases")
        #print(net1.best_testing_biases)
        print(" [",end="")
        for i in range(len(net1.best_testing_biases)):
            print(" np.array(", end="")
            print(net1.best_testing_biases[i],end="")
            print(")",end="")
            if i < len(net1.best_testing_biases)-1:
                print(",",end="")
        print("]")
    elif user == "fwb":
        print("test weights")
        #print(net1.best_training_weights)  
        print(" [",end="")
        for i in range(len(net1.best_training_weights)):
            print(" np.array(", end="")
            print(net1.best_training_weights[i],end="")
            print(")",end="")
            if i < len(net1.best_training_weights)-1:
                print(",",end="")
        print("]")

        print("test biases")
        #print(net1.best_training_biases)
        print(" [",end="")
        for i in range(len(net1.best_training_biases)):
            print(" np.array(", end="")
            print(net1.best_training_biases[i],end="")
            print(")",end="")
            if i < len(net1.best_training_biases)-1:
                print(",",end="")
        print("]")
    
    elif user == "erwb":
        print("test weights")
        #print(net1.lowest_testing_weights)  
        print(" [",end="")
        for i in range(len(net1.lowest_testing_weights)):
            print(" np.array(", end="")
            print(net1.lowest_testing_weights[i],end="")
            print(")",end="")
            if i < len(net1.lowest_testing_weights)-1:
                print(",",end="")
        print("]")

        print("test biases")
        #print(net1.lowest_testing_biases)
        print(" [",end="")
        for i in range(len(net1.lowest_testing_biases)):
            print(" np.array(", end="")
            print(net1.lowest_testing_biases[i],end="")
            print(")",end="")
            if i < len(net1.lowest_testing_biases)-1:
                print(",",end="")
        print("]")
    elif user == "efwb":
        print("test weights")
        #print(net1.lowest_training_weights)  
        print(" [",end="")
        for i in range(len(net1.lowest_training_weights)):
            print(" np.array(", end="")
            print(net1.lowest_training_weights[i],end="")
            print(")",end="")
            if i < len(net1.lowest_training_weights)-1:
                print(",",end="")
        print("]")

        print("test biases")
        #print(net1.lowest_training_biases)
        print(" [",end="")
        for i in range(len(net1.lowest_training_biases)):
            print(" np.array(", end="")
            print(net1.lowest_training_biases[i],end="")
            print(")",end="")
            if i < len(net1.lowest_training_biases)-1:
                print(",",end="")
        print("]")

    elif user == "test":
        i = input()
        testing1 = True
        try:
            val1 = int(i)
            if val1 < 0 or val1 > 99:
                testing1 = False
                print("wtf1")
        except ValueError:
            testing1 = False
            print("wtf2")
        if testing1:
            selected = list_training_data[int(i)]
            print()
            print()
            print(selected[0])
            print(selected[1])
            print(net1.feedforward(np.array(selected[0])))
    elif user == "Tdata":
        print(theTrainingSet[:10])
    user = input()

