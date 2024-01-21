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
                fakeProgressData = self.evaluate(training_data[:1000])
                #print(fakeProgressData[0])
                #print(fakeProgressData[2]/30)
                
                print ("Epoch {0}: {1} / {2}        {3}         {4} / {5}        {6}".format(j+1, realProgressData[0], n_test*5, round(realProgressData[1]/(5*len(test_data)),4), fakeProgressData[0], n_test*5, round(fakeProgressData[1]/(5*len(test_data)),4)))                
                epochsForGraph.append(j)
                training_errors.append(realProgressData[1]/(5*len(test_data)))    
                fakeTraining_errors.append(fakeProgressData[1]/(5*len(test_data)))  
                fakeTraining_inRange.append(fakeProgressData[0])         
                if realProgressData[0] > self.testing_highest:
                    self.testing_highest = realProgressData[0]
                    self.best_testing_biases = self.biases
                    self.best_testing_weights = self.weights
                if fakeProgressData[0] > self.training_highest:
                    self.training_highest = fakeProgressData[0]
                    self.best_training_biases = self.biases
                    self.best_training_weights = self.weights
                
                if realProgressData[1]/(5*len(test_data)) < self.testing_lowest_error:
                    
                    self.testing_lowest_error = realProgressData[1]/(5*len(test_data))
                    self.lowest_testing_biases = self.biases
                    self.lowest_testing_weights = self.weights
                    
                else:#not sure if this is how its supposed to work, but bweh
                    self.biases = self.lowest_testing_biases
                    self.weights = self.lowest_testing_weights
                    #print(round(self.testing_lowest_error,4),round(eta,5),sep="   ")
                    #eta = random.uniform(-1, 1)
                    
                    #eta = eta * (10/100)

                if fakeProgressData[1]/(5*len(test_data)) < self.training_lowest_error:
                    self.training_lowest_error = fakeProgressData[1]/(5*len(test_data))
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
            nabla_b[-l] = clip_gradients(delta,1000000)
            #nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_w[-l] = clip_gradients(np.dot(delta, activations[-l-1].transpose()), 1000000)
            

        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives for the output activations."""      
        
        return (output_activations-y)

def clip_gradients(grad, threshold):
    #print("gradient clipped to:", threshold,sep=" ")
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

theTrainingSet = get_training_data5(1001000)


list_training_data = []
test_data = []
for i in range(len(theTrainingSet)):
    if i < (len(theTrainingSet) - 1000):
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

new_bisesER0 = [np.array([[-0.39079109],
       [-0.08994521],
       [ 0.04637683],
       [ 1.5241424 ],
       [-0.62461999],
       [-1.90620744],
       [-0.76501173],
       [-0.18389033],
       [ 1.50181057],
       [ 0.12784717],
       [ 0.37333843],
       [-0.46054382],
       [-0.10169387],
       [-0.10348016],
       [-1.86116569],
       [ 1.87362841]]), np.array([[-0.31951723],
       [-0.23045932],
       [-0.35491858],
       [-0.08224848],
       [ 0.99536523],
       [ 1.50089535],
       [-0.59048271],
       [-0.82124547],
       [ 0.6514802 ],
       [ 0.29369273],
       [-0.57937986],
       [-0.23390633],
       [-1.37730357],
       [-1.08485094],
       [ 0.21443272],
       [-1.2205252 ]]), np.array([[-0.59913739],
       [ 1.03699991],
       [-0.72221992],
       [ 3.48631647],
       [-0.54182627],
       [ 0.75700076],
       [ 0.58809494],
       [-0.83143986],
       [ 1.14620923],
       [ 2.11703771],
       [-0.8791864 ],
       [-1.86331615],
       [ 1.5450031 ],
       [-0.85595761],
       [ 1.99727489],
       [ 1.32072223]]), np.array([[-1.30872531],
       [-0.87855401],
       [ 0.95069978],
       [ 0.57655801],
       [ 1.40576716]])]
new_weightsER0 = [np.array([[ 1.78436732e-01,  1.51645015e-01,  4.41484977e-01,
        -1.27773262e-02,  4.64799864e-01],
       [-1.17313616e-01, -7.04715057e-01,  9.88301941e-01,
         1.98192856e-01,  4.84618038e-01],
       [ 8.33662394e-01, -1.09903527e-01, -1.44866309e-01,
        -8.95649824e-01,  1.21586936e+00],
       [ 8.77741631e-02,  4.09550726e-01, -3.17632579e-01,
         2.69597121e-01,  6.69276062e-01],
       [-1.59468684e-01, -1.30226521e-01, -7.99070844e-01,
         1.88998735e-02, -1.01947639e+00],
       [-8.00921462e-01, -4.43296429e-01,  2.45667065e-01,
        -1.02811477e+00,  5.26172422e-01],
       [-1.69965757e-01,  1.39771852e-01,  3.26266631e-01,
        -4.55324542e-01,  1.76877191e-01],
       [ 3.71585426e-01, -7.48358108e-02, -2.53513614e-01,
         7.73900263e-01,  5.78028786e-01],
       [ 8.40353990e-01,  4.43745788e-01,  1.44304504e-03,
         2.16830553e-01, -1.53196412e-01],
       [-7.90490480e-01,  5.83478095e-01,  1.00802791e+00,
         8.28266567e-01,  5.37135196e-01],
       [ 2.67845394e-01,  5.23755701e-01,  1.05135746e+00,
         3.75349247e-01,  1.39713552e+00],
       [-1.10119644e-01,  5.85937879e-01,  8.12716601e-03,
         1.43357712e+00, -3.27148114e-01],
       [ 1.33919315e+00,  1.12144785e+00, -2.97543114e-01,
         7.17320443e-01,  9.42071146e-01],
       [-4.52573864e-01,  2.84787392e-02, -1.04240145e+00,
        -4.73470607e-01,  5.35748786e-01],
       [-4.91210524e-01,  9.80042095e-01,  2.01208799e+00,
        -6.63717595e-01, -1.13594227e-01],
       [ 1.33594554e-01, -9.84674741e-03, -1.80634969e-01,
        -6.01748474e-01, -6.24305602e-01]]), np.array([[-0.2616812 , -0.42691681, -0.16102374, -0.42113763, -0.40504799,
        -0.20336369,  0.57478884, -0.18902238, -0.45701932,  0.08373543,
         0.26681708,  0.04599604,  0.13127785,  0.04945773,  0.01350133,
        -0.14133936],
       [-0.31683248,  0.14639726,  0.43458642,  0.22564836, -0.42329016,
        -0.43904583, -0.24101635,  0.28150139,  0.11438863, -0.21498702,
        -0.68825244,  0.33241494,  0.21472389, -0.32239175, -0.47037115,
        -0.21891101],
       [-0.01832294, -0.38161219,  0.11396591, -0.0396837 ,  0.05851557,
        -0.23628914, -0.00572741, -0.21014288, -0.41762396,  0.03638068,
        -0.36165293,  0.21364524, -0.81025515, -0.02626079, -0.03055265,
         0.56825364],
       [ 0.37347473,  0.69845781,  0.95557599,  0.15404217,  0.01515547,
         1.05605533,  0.16737382,  0.21833228,  0.22413622, -0.15165292,
         0.67081458, -0.2715662 ,  0.10081946,  0.47320604,  1.32743286,
        -0.02480304],
       [-0.05307994, -0.15352439, -0.95443979, -0.17566028,  0.046886  ,
         0.09086584, -0.03884431,  0.40022599, -0.24174046, -0.55924654,
         0.2843178 ,  0.25038122,  0.17223021, -0.19800937, -0.45598748,
        -0.32191972],
       [ 0.19864954, -0.36788821,  0.20226324,  0.17460561,  0.12940962,
        -0.38546183,  0.3473339 ,  0.13099835, -0.26497613, -0.3826109 ,
         0.29972783, -0.32932473, -0.12758951,  0.17662236, -0.26308628,
        -0.73269855],
       [ 0.28421752, -0.25336673, -0.11406219,  0.52386174, -0.3390438 ,
         0.39186019, -0.1982446 , -0.84451667, -0.07147366,  0.43741112,
        -0.06600104,  0.73851547,  0.49970939, -0.0872377 , -0.59454235,
         0.35911167],
       [ 0.19681709, -0.44063848,  0.03990689, -0.2006285 ,  0.28974492,
        -0.67841469, -0.34758093, -0.40776364,  0.15687317, -0.43471713,
        -0.69907575,  0.31235384,  0.30808286, -0.1373946 , -0.08005303,
         0.22737818],
       [-0.02882179, -0.57185277,  0.02019618,  0.28671025, -0.11658827,
        -0.82637397,  0.19793622,  0.31052313, -0.27791003,  0.69371079,
         0.3405668 ,  1.12175523, -0.21557209,  0.68698492,  0.13565035,
        -0.00873457],
       [-0.57240945, -0.5042591 ,  0.26992908, -0.37423017, -0.26719719,
         0.16100266, -0.76480645, -0.66742343, -0.2670832 , -0.3910146 ,
        -0.14618965,  0.31525526,  0.32771908,  0.24169837, -0.1377743 ,
         0.29701504],
       [-0.06194603, -0.19463867, -0.75571527, -0.16057235,  0.21907907,
        -0.55173193, -0.28654532, -0.28340205,  0.41844629,  0.99891513,
        -0.08776657,  0.17705702, -0.13199143, -0.30044047,  0.3319557 ,
        -0.47696194],
       [-0.56451235, -0.81248831,  0.07033799,  0.25350336, -0.70972715,
         0.37715689, -0.21422116, -0.0682689 , -0.32164332,  0.17557233,
        -0.25314353,  0.17676686, -0.04268194, -0.45720773, -0.23695242,
         0.41406306],
       [ 0.19425772, -0.35908095, -0.29107907,  0.51165066, -0.41204344,
        -0.62653143, -0.67076883, -0.14129206,  0.48551369,  0.31942595,
         0.58983107, -0.021021  ,  0.05880523,  0.02955781,  0.31516163,
        -0.11610938],
       [ 0.23004564,  0.10651008, -0.07463991, -0.44889385, -0.78800952,
        -0.19498572,  0.25738595, -0.48519064, -0.2169263 , -0.10266549,
         0.14734272, -0.06420536, -0.11567747, -0.20452089, -0.26050549,
        -0.21418438],
       [ 0.12903302, -1.11870397,  0.34578363,  0.48090482, -0.85445027,
         0.16650631, -0.26438515,  0.31441971, -0.38615717, -0.01522818,
        -0.62732533,  0.57194082, -0.39942499, -0.52197933,  0.57289554,
         0.00958873],
       [-0.21912647, -1.05811728, -0.07053081,  0.03256696, -0.42487729,
         0.22310247,  0.0554331 , -0.24553075,  0.04374073,  0.55088206,
         0.46285987,  0.27453662,  0.03664891,  0.52055627, -0.42805288,
         0.00352151]]), np.array([[ 8.44941425e-02, -2.73929503e-01, -3.78804595e-01,
        -7.52611364e-01, -7.33404394e-02, -1.81205435e-01,
        -4.50159533e-01, -8.47304001e-02,  1.75801892e-01,
         1.80968540e-01,  1.56612481e-01, -4.75672735e-02,
        -6.55414313e-02,  5.72775736e-01,  3.88451781e-01,
        -4.10306765e-02],
       [-8.46494162e-01,  5.86173197e-01,  6.02615325e-03,
        -8.66649942e-01, -3.43488146e-02, -5.71621571e-02,
        -1.35279720e-01, -8.52615164e-02, -7.95985312e-02,
         2.88489543e-01,  2.31217159e-01, -3.19745518e-02,
         2.28385204e-01,  6.78181098e-01, -2.72528737e-01,
        -1.58461676e-01],
       [-9.17716358e-01, -3.07254315e-01,  3.66048022e-01,
        -6.40893655e-01, -3.87097278e-01,  8.78178639e-03,
        -2.41551738e-01, -2.79588540e-01,  5.83624990e-01,
        -2.43645662e-01, -6.71925165e-02,  1.61435018e-01,
        -6.42316755e-02, -5.16155404e-01, -7.20548955e-01,
         1.91230218e-01],
       [-4.27283615e-01, -2.59255151e-01,  3.03406924e-01,
        -1.64540478e-01,  6.51379937e-01,  1.90826258e-01,
         1.31160338e-01,  1.84753373e-01, -7.37204003e-02,
        -1.27618664e-01, -8.26147119e-01, -2.39860536e-01,
        -3.96019874e-01,  2.63319773e-01, -2.67464688e-01,
        -3.52171181e-01],
       [-5.09412709e-02,  1.38544240e-01, -4.64925719e-01,
        -3.69694990e-01, -5.16688013e-02, -1.87273337e-01,
        -2.00179289e-01, -9.53053724e-02,  6.62351467e-01,
        -3.69744988e-01,  2.55682650e-01,  6.07577227e-01,
         2.05919352e-01,  8.91163038e-02,  2.17315253e-01,
         1.52714689e-01],
       [-7.59896410e-01, -5.87632821e-01, -8.76474080e-04,
        -7.22822800e-01,  1.61577638e-01, -3.69825416e-01,
        -2.64407064e-01, -5.28666517e-02,  1.72644948e-01,
         6.59381075e-01, -2.29410586e-01,  5.75218068e-01,
        -3.14176266e-01,  2.37400559e-01,  4.24586206e-01,
         1.85292004e-01],
       [-7.20681050e-01,  6.15918457e-01,  5.15005176e-02,
        -6.54443617e-01, -3.24870705e-01,  3.69517665e-01,
        -6.43422115e-02,  1.45391746e-01,  6.80032602e-02,
         1.28166726e-01,  1.54573923e-01,  1.86851861e-01,
        -4.09678114e-01, -5.68012683e-01, -3.59605688e-01,
        -1.88782969e-01],
       [ 3.81576230e-01, -1.07948838e-01,  2.98759180e-01,
        -4.85512470e-01,  7.20525982e-01, -5.51112851e-01,
         1.88102718e-01, -1.42861472e-02,  3.09243222e-01,
         8.32615154e-02, -3.63031742e-01, -2.53908451e-01,
        -5.49772060e-01,  6.49387060e-01,  3.08204512e-02,
         3.71230868e-01],
       [-3.92232250e-01, -7.53275193e-02, -6.07788591e-01,
        -9.07751352e-01,  4.18132038e-01,  3.27797364e-02,
        -2.52661429e-01,  9.95806354e-03,  1.72641363e-01,
        -3.14826926e-01,  1.21864043e-01,  6.18289863e-03,
        -2.13904729e-01,  3.98109501e-01, -2.37724717e-01,
         2.66551493e-01],
       [-8.63426967e-02,  2.34610185e-01, -7.43572681e-02,
        -1.23556371e+00,  3.93763534e-01, -1.31508329e-02,
         3.92511831e-01, -3.26453943e-01,  3.23486526e-02,
        -8.19500016e-01,  3.06838686e-01,  4.14901953e-01,
        -1.11201205e-01, -2.77193979e-01, -4.07966350e-01,
         2.60044075e-01],
       [-3.36299768e-01,  2.97430065e-01, -1.66344018e-01,
        -8.53089765e-01,  5.08835192e-01, -4.32183804e-01,
         9.44994218e-03,  3.49368320e-01,  2.79469300e-01,
        -4.86854862e-01, -3.90412036e-01,  1.82894538e-01,
        -1.41689355e-02, -1.80566974e-01, -1.97131915e-01,
         5.91519408e-01],
       [ 3.65639342e-01, -3.28207255e-01,  3.15970522e-01,
        -1.19228165e+00, -4.38614982e-02, -9.64746136e-02,
         3.83603392e-01,  5.75252300e-01,  2.77799998e-02,
         3.98135588e-01,  3.09843274e-01,  2.46949398e-01,
        -4.18601386e-01,  6.37385562e-02, -3.18028815e-01,
        -9.16896992e-01],
       [-1.38252114e-02, -2.66021412e-01, -4.33642929e-01,
        -1.12021951e+00, -5.64967237e-01, -2.35915095e-01,
         6.41112429e-03,  7.26689714e-01,  4.95399643e-01,
         2.99865524e-01, -9.20020515e-02,  2.28614819e-02,
        -5.45102188e-01,  3.45885942e-01, -3.10825858e-01,
         1.19260365e-02],
       [ 5.13214618e-01, -7.11630266e-01, -2.43650239e-01,
        -6.14779125e-01, -3.02692021e-01,  6.91067571e-02,
         2.73254518e-01,  2.93819193e-01, -1.91652495e-01,
        -4.08658067e-01,  7.38766031e-02,  1.34351782e-01,
        -1.61543536e-01, -4.01345554e-01, -4.55316709e-01,
         7.47665177e-01],
       [ 3.60164735e-01,  1.19626416e-01,  4.02258203e-01,
        -7.23955156e-01, -6.36890497e-01, -3.53741150e-02,
        -2.01551680e-01,  5.51334689e-02, -5.28704195e-01,
         7.57631359e-02, -4.02889449e-01,  6.73860752e-02,
         2.66136934e-02, -2.49549704e-01, -7.00406401e-01,
        -3.42375963e-01],
       [-3.11647751e-02, -4.08940811e-02,  7.89234011e-01,
        -9.93630005e-01, -1.93746727e-01,  3.35752706e-01,
        -2.24714905e-01, -2.91205474e-01,  3.51348263e-01,
        -2.59569978e-01, -2.08309100e-01, -2.69877967e-01,
        -3.86525965e-02,  2.09103719e-01,  3.02255065e-02,
         4.51765530e-02]]), np.array([[ 0.3031209 , -0.33501927,  0.41262558, -0.30301283, -1.17724111,
         0.14363585, -0.60341599, -0.37017051, -0.32353779, -0.41652317,
        -0.2844299 , -0.33418826, -0.41245776, -0.82463762, -0.99319137,
         0.32736013],
       [-1.13504192, -0.59771184,  0.01203768,  0.0298242 , -0.50869081,
        -0.50593236, -0.24161548, -0.49501334,  0.10760718, -0.45832334,
        -1.04241852, -0.4162179 , -0.48379237, -0.04700851,  1.19037027,
        -0.40133113],
       [-0.42766444,  0.27657664,  0.23841154,  0.19555432,  0.13769407,
         0.02264831, -0.70986488, -0.61156726,  0.3016097 ,  0.1397683 ,
        -0.72361738, -0.43643074, -0.35468876, -0.47104495,  0.92187559,
         0.35750467],
       [-0.05172309, -0.07159166,  0.85026917,  0.37811924, -0.14286882,
        -0.29493863, -0.50320552,  0.51381141, -0.18568059,  0.24985602,
        -0.06579304, -0.46877388,  0.3174728 , -0.21006354,  0.93111311,
        -0.34615191],
       [-0.14725277,  0.01115455, -0.12351874,  0.33180754, -0.15994871,
         0.17503456,  0.48879343, -0.04354568,  0.20393979,  0.00217351,
        -0.10885819, -0.48450643,  0.24311021,  0.91921282,  0.93303303,
        -0.82601896]])]

new_bisesER1 = [np.array([[-0.21771281],
       [ 0.19530784],
       [ 0.36782077],
       [ 1.91774753],
       [-0.62176746],
       [-1.96960677],
       [-0.76513212],
       [-0.1810378 ],
       [ 1.86057277],
       [ 0.10479064],
       [ 0.70052632],
       [-0.45769129],
       [-0.09884134],
       [-0.10062763],
       [-2.00062911],
       [ 2.15919791]]), np.array([[-0.31652526],
       [-0.24170923],
       [-0.35192661],
       [ 0.21679763],
       [ 1.21298252],
       [ 1.59974635],
       [-0.13095794],
       [-0.8182535 ],
       [ 1.09457347],
       [ 0.2966847 ],
       [-0.57638789],
       [-0.25895465],
       [-1.39662149],
       [-1.08185897],
       [ 0.21024325],
       [-1.21753323]]), np.array([[-0.60948655],
       [ 1.00815317],
       [-0.73256908],
       [ 3.60591359],
       [-0.55217543],
       [ 0.77123752],
       [ 0.11739331],
       [-0.84178902],
       [ 1.0963966 ],
       [ 1.64952278],
       [-0.88953556],
       [-1.87366531],
       [ 0.9200281 ],
       [-0.86630677],
       [ 1.79819283],
       [ 1.21977994]]), np.array([[-0.79356282],
       [-0.06083609],
       [ 1.10049931],
       [ 0.89116156],
       [ 1.64126465]])]
new_weightsER1 = [np.array([[ 2.87298849e-01,  1.63282548e-01,  1.44389324e-01,
        -8.43064847e-02,  7.70450485e-01],
       [ 1.14184492e-01, -8.42048600e-01,  2.71549586e-01,
         1.76311410e-01,  1.12675244e+00],
       [ 1.06515202e+00, -3.02504086e-01, -8.37097924e-01,
        -9.20286176e-01,  1.85787165e+00],
       [ 3.05439260e-01, -1.11692870e-01, -5.98836509e-01,
         8.39741884e-02,  1.29861907e+00],
       [-1.57153703e-01, -1.31599856e-01, -8.06238368e-01,
         1.86810590e-02, -1.01305505e+00],
       [-7.36975627e-01,  3.88530961e-02, -4.00972941e-01,
        -8.85162092e-01,  6.83835242e-01],
       [-1.14947148e-01,  3.64295193e-01, -1.33124895e-02,
        -4.31525112e-01,  3.28065582e-01],
       [ 3.73900407e-01, -7.62091462e-02, -2.60681138e-01,
         7.73681449e-01,  5.84450130e-01],
       [ 1.06417956e+00,  4.03796027e-02, -3.80233020e-01,
         3.13360825e-02,  5.01908586e-01],
       [-7.80836533e-01,  6.83102475e-01,  9.01847074e-01,
         8.37788050e-01,  5.67270924e-01],
       [ 4.99342571e-01,  3.34823711e-01,  3.50229914e-01,
         3.52154379e-01,  2.03928740e+00],
       [-1.07804663e-01,  5.84564544e-01,  9.59642457e-04,
         1.43335831e+00, -3.20726770e-01],
       [ 1.34150813e+00,  1.12007451e+00, -3.04710638e-01,
         7.17101629e-01,  9.48492490e-01],
       [-4.50258883e-01,  2.71054038e-02, -1.04956897e+00,
        -4.73689421e-01,  5.42170130e-01],
       [-4.35913672e-01,  1.65826568e+00,  1.15927619e+00,
        -3.99469100e-01, -1.12107487e-03],
       [ 3.65119073e-01, -1.48420243e-01, -8.95535971e-01,
        -6.24904881e-01,  1.82023216e-02]]), np.array([[-0.26136775, -0.43807024, -0.1477407 , -0.41280795, -0.40506439,
        -0.20389038,  0.57505412, -0.18893658, -0.45454021,  0.08314079,
         0.26744429,  0.04595904,  0.13145124,  0.04961481,  0.00626378,
        -0.14011083],
       [-0.3164283 ,  0.11777034,  0.44613689,  0.21526987, -0.42318641,
        -0.43930899, -0.24059191,  0.28162096,  0.10061242, -0.21551637,
        -0.68833667,  0.33257138,  0.21516707, -0.32214165, -0.47714082,
        -0.24094927],
       [-0.01800949, -0.39276562,  0.12724895, -0.03135402,  0.05849917,
        -0.23681583, -0.00546213, -0.21005708, -0.41514485,  0.03578604,
        -0.36102572,  0.21360824, -0.81008176, -0.02610371, -0.0377902 ,
         0.56948217],
       [ 0.40482034, -0.41740348,  2.28387971,  0.9869478 ,  0.01351973,
         1.00338955,  0.19390366,  0.22691445,  0.4719604 , -0.21111714,
         0.73345568, -0.27526032,  0.11816623,  0.48891815,  0.60368259,
         0.09796411],
       [-0.05480385, -0.21608175, -1.05894021, -0.02531025,  0.04581479,
         0.08594089, -0.04147781,  0.40122662, -0.19967405, -0.56160162,
         0.05044403,  0.24830754,  0.17026035, -0.19763008, -0.47357948,
        -0.19958035],
       [ 0.16293006, -0.5104238 ,  0.88747757,  0.48599297,  0.1267686 ,
        -0.35404178,  0.36726167,  0.1341607 , -0.26058905, -0.38625873,
         0.45773906, -0.33329934, -0.12273607,  0.18157449, -0.26234667,
        -0.79513021],
       [ 0.25840779, -0.26698287,  0.75631058,  1.226077  , -0.34504044,
         0.36660827, -0.19354598, -0.84013119,  0.11358606,  0.43073627,
        -0.00426077,  0.72908329,  0.50146285, -0.08145498, -0.82531673,
         0.6424673 ],
       [ 0.19713054, -0.45179191,  0.05318993, -0.19229882,  0.28972852,
        -0.67894138, -0.34731565, -0.40767784,  0.15935228, -0.43531177,
        -0.69844854,  0.31231684,  0.30825625, -0.13723752, -0.08729058,
         0.22860671],
       [-0.05618828, -0.57170674,  0.82625355,  0.94121263, -0.12238175,
        -0.84641474,  0.20627807,  0.31480669, -0.1226151 ,  0.68722042,
         0.3755381 ,  1.11266238, -0.21424275,  0.69232534, -0.05269247,
         0.23576089],
       [-0.572096  , -0.51541253,  0.28321212, -0.36590049, -0.26721359,
         0.16047597, -0.76454117, -0.66733763, -0.26460409, -0.39160924,
        -0.14556244,  0.31521826,  0.32789247,  0.24185545, -0.14501185,
         0.29824357],
       [-0.06163258, -0.2057921 , -0.74243223, -0.15224267,  0.21906267,
        -0.55225862, -0.28628004, -0.28331625,  0.4209254 ,  0.99832049,
        -0.08713936,  0.17702002, -0.13181804, -0.30028339,  0.32471815,
        -0.47573341],
       [-0.56406792, -0.83140018,  0.08042199,  0.21885464, -0.70956882,
         0.37715399, -0.21371333, -0.06813492, -0.3591261 ,  0.17501978,
        -0.25889136,  0.17693588, -0.0423294 , -0.45699752, -0.24350582,
         0.36363127],
       [ 0.19466083, -0.37716516, -0.28374052,  0.48109824, -0.41192335,
        -0.62661839, -0.67030927, -0.14116845,  0.45131555,  0.31885073,
         0.58084175, -0.02091219,  0.05908574,  0.02974855,  0.30847372,
        -0.15957882],
       [ 0.23035909,  0.09535665, -0.06135687, -0.44056417, -0.78802592,
        -0.19551241,  0.25765123, -0.48510484, -0.21444719, -0.10326013,
         0.14796993, -0.06424236, -0.11550408, -0.20436381, -0.26774304,
        -0.21295585],
       [ 0.12937817, -1.13101533,  0.3582737 ,  0.47798553, -0.85442271,
         0.16611482, -0.2640602 ,  0.31451777, -0.39423887, -0.01581683,
        -0.62866294,  0.5719504 , -0.39921792, -0.52181155,  0.56582256,
        -0.00257068],
       [-0.21881302, -1.06927071, -0.05724777,  0.04089664, -0.42489369,
         0.22257578,  0.05569838, -0.24544495,  0.04621984,  0.55028742,
         0.46348708,  0.27449962,  0.0368223 ,  0.52071335, -0.43529043,
         0.00475004]]), np.array([[ 8.48017859e-02, -2.73760125e-01, -3.78694412e-01,
        -1.01069373e+00, -6.70465068e-02, -2.08621509e-01,
        -4.88960495e-01, -8.41762901e-02,  1.42305751e-01,
         1.81016639e-01,  1.57681852e-01, -4.75722662e-02,
        -6.54196002e-02,  5.73164060e-01,  3.88040110e-01,
        -4.11081568e-02],
       [-8.45804600e-01,  5.86375632e-01,  6.28854642e-03,
        -1.14563278e+00, -2.91608172e-02, -8.45531453e-02,
        -1.76297274e-01, -8.43832038e-02, -1.11652083e-01,
         2.88693828e-01,  2.32611343e-01, -3.16469981e-02,
         2.28771796e-01,  6.78869214e-01, -2.72530641e-01,
        -1.57935241e-01],
       [-9.17408715e-01, -3.07084937e-01,  3.66158205e-01,
        -8.98976022e-01, -3.80803345e-01, -1.86342880e-02,
        -2.80352700e-01, -2.79034430e-01,  5.50128849e-01,
        -2.43597563e-01, -6.61231458e-02,  1.61430025e-01,
        -6.41098444e-02, -5.15767080e-01, -7.20960626e-01,
         1.91152738e-01],
       [-4.39943921e-01, -2.83429130e-01,  2.59572156e-01,
         1.31635583e-01,  1.02981876e+00, -6.47902534e-01,
        -1.03917899e+00,  1.25398309e-01, -1.12163454e+00,
        -1.73434851e-01, -8.25261055e-01, -3.28414589e-01,
        -4.34090412e-01,  2.90435706e-01, -3.87143951e-01,
        -4.53529887e-01],
       [-5.06336275e-02,  1.38713618e-01, -4.64815536e-01,
        -6.27777357e-01, -4.53748687e-02, -2.14689411e-01,
        -2.38980251e-01, -9.47512624e-02,  6.28855326e-01,
        -3.69696889e-01,  2.56752021e-01,  6.07572234e-01,
         2.06041183e-01,  8.95046276e-02,  2.16903582e-01,
         1.52637209e-01],
       [-7.60064973e-01, -5.87455216e-01, -7.77567489e-04,
        -9.61369168e-01,  1.66591020e-01, -3.97209131e-01,
        -2.76640397e-01, -5.24719637e-02,  1.55780870e-01,
         6.59365917e-01, -2.28637544e-01,  5.75567681e-01,
        -3.14161736e-01,  2.37224608e-01,  4.24249087e-01,
         1.84801439e-01],
       [-7.10206175e-01,  6.16724448e-01,  6.74368674e-02,
        -1.06386139e+00, -4.50426245e-01,  3.45918850e-01,
         6.66896843e-01,  1.69272605e-01,  6.02205512e-01,
         1.41557514e-01,  1.67799983e-01,  2.16953027e-01,
        -3.89160147e-01, -5.70427238e-01, -3.21623277e-01,
        -1.51401507e-01],
       [ 3.81883873e-01, -1.07779460e-01,  2.98869363e-01,
        -7.43594837e-01,  7.26819915e-01, -5.78528925e-01,
         1.49301756e-01, -1.37320372e-02,  2.75747081e-01,
         8.33096139e-02, -3.61962371e-01, -2.53913444e-01,
        -5.49650229e-01,  6.49775384e-01,  3.04087798e-02,
         3.71153388e-01],
       [-3.91090175e-01, -7.50843629e-02, -6.07251391e-01,
        -1.21074284e+00,  4.18941551e-01,  5.50755673e-03,
        -2.93084576e-01,  1.13344426e-02,  1.44798791e-01,
        -3.14349498e-01,  1.23644745e-01,  7.35030557e-03,
        -2.13102052e-01,  3.99051794e-01, -2.36904187e-01,
         2.67981674e-01],
       [-7.99784839e-02,  2.33769030e-01, -6.85190110e-02,
        -1.57095617e+00,  5.97230426e-01, -5.15575383e-02,
         1.29104311e+00, -3.14347259e-01,  6.14055650e-01,
        -8.15889966e-01,  3.18506620e-01,  4.28396191e-01,
        -9.61866810e-02, -2.74408964e-01, -3.94337432e-01,
         2.80128125e-01],
       [-3.35992125e-01,  2.97599443e-01, -1.66233835e-01,
        -1.11117213e+00,  5.15129125e-01, -4.59599878e-01,
        -2.93510196e-02,  3.49922430e-01,  2.45973159e-01,
        -4.86806763e-01, -3.89342665e-01,  1.82889545e-01,
        -1.40471044e-02, -1.80178650e-01, -1.97543586e-01,
         5.91441928e-01],
       [ 3.65946985e-01, -3.28037877e-01,  3.16080705e-01,
        -1.45036402e+00, -3.75675656e-02, -1.23890688e-01,
         3.44802430e-01,  5.75806410e-01, -5.71614077e-03,
         3.98183687e-01,  3.10912645e-01,  2.46944405e-01,
        -4.18479555e-01,  6.41268800e-02, -3.18440486e-01,
        -9.16974472e-01],
       [-6.34459462e-04, -2.65299902e-01, -4.16725275e-01,
        -1.65121901e+00, -6.55525673e-01, -2.58969593e-01,
         8.13069312e-01,  7.53245741e-01,  1.11410692e+00,
         3.14180498e-01, -7.57088426e-02,  5.65850287e-02,
        -5.21567134e-01,  3.45548859e-01, -2.69399917e-01,
         5.46893149e-02],
       [ 5.13522261e-01, -7.11460888e-01, -2.43540056e-01,
        -8.72861492e-01, -2.96398088e-01,  4.16906827e-02,
         2.34453556e-01,  2.94373303e-01, -2.25148636e-01,
        -4.08609968e-01,  7.49459738e-02,  1.34346789e-01,
        -1.61421705e-01, -4.00957230e-01, -4.55728380e-01,
         7.47587697e-01],
       [ 3.62843546e-01,  1.16826099e-01,  3.96213578e-01,
        -6.62156497e-01, -4.44755840e-01, -5.87629659e-02,
        -3.53356604e-01,  4.77113060e-02, -6.00137960e-01,
         7.08100997e-02, -4.02324165e-01,  5.53922435e-02,
         2.30450823e-02, -2.39847859e-01, -7.20024354e-01,
        -3.51357659e-01],
       [-2.89272168e-02, -4.05860797e-02,  7.90384495e-01,
        -1.35220048e+00, -1.90142279e-01,  3.08719061e-01,
        -2.63375205e-01, -2.88640004e-01,  3.39999403e-01,
        -2.58469514e-01, -2.05523946e-01, -2.66942309e-01,
        -3.68247012e-02,  2.10679077e-01,  3.29081508e-02,
         4.88020356e-02]]), np.array([[ 0.11773681, -0.53337041,  0.26686187, -0.60338603, -1.27074674,
        -0.02345618, -0.77932482, -0.49291794, -0.52941193, -0.6486288 ,
        -0.48230843, -0.59340631, -0.73823744, -0.96912539, -0.99093758,
         0.10726716],
       [-1.30305843, -0.77100567, -0.11767651,  0.87212943, -0.59195619,
        -0.6524937 , -0.4321094 , -0.6063036 , -0.07114836, -0.17266625,
        -1.22035535, -0.64952329, -0.75463328, -0.17673132,  1.67925568,
        -0.59044947],
       [-0.53010169,  0.16389496,  0.15666841,  0.4637748 ,  0.08439779,
        -0.07308089, -0.93652413, -0.68335026,  0.1841808 , -0.13391847,
        -0.83663052, -0.58285618, -0.71043115, -0.55155281,  1.0987757 ,
         0.23109403],
       [-0.1940373 , -0.2261681 ,  0.73635512,  0.61725716, -0.21654652,
        -0.42561506, -0.75784223,  0.41576266, -0.34666501, -0.11372908,
        -0.22172579, -0.67293744, -0.09714543, -0.322496  ,  1.04395664,
        -0.51960213],
       [-0.24595739, -0.09585696, -0.20266717,  0.49103969, -0.2110695 ,
         0.0846173 ,  0.31828452, -0.11150599,  0.09242767, -0.23761963,
        -0.21707187, -0.62655848, -0.03339632,  0.84103066,  0.98971653,
        -0.94636798]])]


net1 = Network([5, 16,16,16, 5])
net1.biases = new_bisesER1
net1.weights = new_weightsER1
net1.SGD(list_training_data, 10, 10000, 1, test_data=test_data)

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
        print(net1.lowest_testing_weights)  
        """print(" [",end="")
        for i in range(len(net1.lowest_testing_weights)):
            print(" np.array(", end="")
            print(net1.lowest_testing_weights[i],end="")
            print(")",end="")
            if i < len(net1.lowest_testing_weights)-1:
                print(",",end="")
        print("]")"""

        print("test biases")
        print(net1.lowest_testing_biases)
        """print(" [",end="")
        for i in range(len(net1.lowest_testing_biases)):
            print(" np.array(", end="")
            print(net1.lowest_testing_biases[i],end="")
            print(")",end="")
            if i < len(net1.lowest_testing_biases)-1:
                print(",",end="")
        print("]")"""
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
