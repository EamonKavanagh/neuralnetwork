A neural network implementation in python 2.7 that requires numpy.  Currently, the network only really supports digit classification for the MNIST dataset and one hidden layer but it should be easy to generalize.  

The Network class expects a training set.  This should be a numpy array where each row in the array is an observation from the training set.  Network also expects the corresponding digits that each observation belongs to which is a one dimensional numpy array, one for each observation.  You also need to specify the number of hidden nodes.

Network has two main functions: one to train the network with either gradient descent, incremental gradient, or a hybrid method (http://stuff.mit.edu/people/dimitrib/incrgrad.pdf).  Parameters unique to each method are required.  The other is a predict function that takes in new data in a similar format to the training set and outputs the predicted digits.  You may also check the cost function value.  The checkDeriv function is for debugging purposes.

I have included a file to test it out (demo.py) as well as a dataset that can be run to see it in action.  To run, navigate to the directory containing the files and type 'python demo.py XX'  where XX is replaced by either GD (gradient descent), ID (incremental gradient descent), or HD (hybrid method).
