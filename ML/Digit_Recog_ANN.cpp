#include <stdio.h> 
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <iostream> 
#include "data_parser.h"
#include <string.h>
class Network { 

	public: 
	int num_layers; 
	int* sizes; 
	gsl_vector* biases; 
	gsl_matrix*  weights;
	

	Network (int *node_layers, int num_layers) 
	{
	this->num_layers = num_layers; 
	this->sizes = node_layers; 
	this->biases = (gsl_vector*) malloc((num_layers - 1) * sizeof(gsl_vector));
	this->weights = (gsl_matrix*) malloc((num_layers - 1) * sizeof(gsl_matrix)); 

	gsl_rng *rng = gsl_rng_alloc(gsl_rng_rand); 
	gsl_rng_set(rng, 8080); 
	// Instantiate bias vectors
	for (int i = 1; i < num_layers; i++) {
		int nodes = *(sizes + i); 
		gsl_vector *bias_layer = gsl_vector_alloc(nodes); 
		for (int j = 0; j < nodes; j++) {
			gsl_vector_set(bias_layer, j, gsl_rng_uniform_pos(rng)); 
		}
		*(this->biases + i - 1) = *bias_layer;
	}

	// Instantiate weight vectors
	for (int i = 0; i < num_layers - 1; i++ ) {
		int layer_0 = *(sizes + i); // x 
		int layer_1 = *(sizes + i + 1); //y
		gsl_matrix *weight_layer = gsl_matrix_alloc(layer_1, layer_0);
		for (int j = 0; j < layer_0; j++ ) {
			for (int k = 0; k < layer_1; k++) {
				gsl_matrix_set(weight_layer, k, j, gsl_rng_uniform_pos(rng)); 
			}
		*(this->weights + i) = *weight_layer; 
		}
	}
	// `gsl_rng_free(rng); 
	}

	/** 
	def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
	
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
*/


	/**~Network () 
	{

	}*/

	void print_biases(void){
	
		std::cout << num_layers << "\n"; 
		std::cout << "cout\n"; 
		for(int j = 0; j < num_layers - 1; j++) {
			for (int i = 0; i < 3; i++) {
				std::cout << gsl_vector_get(&biases[j], i) << "\n";
			}
		}
		std::cout << "fprintf\n"; 
		gsl_vector_fprintf(stdout, biases, "%f"); //astd::cout << *(this->biases) << "\n"; 
		gsl_vector_fprintf(stdout, &biases[1], "%f"); //astd::cout << *(this->biases) << "\n"; 

	}

	
};

int main(int argc, char *argv[]) {
	int* layers = (int* ) malloc(sizeof(int) *3); // 
	*layers = 5; 
	*(layers + 1) = 12; 
	*(layers + 2) = 3; 
	


	Network *net = new Network(layers, 3); 
	number_image_pairs* pairs = Get_Number_Images_From_Idx("train_images", "train_labels"); 
	
	

}

