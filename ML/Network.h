#include <stdio.h> 
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <iostream> 
#include <string.h>
#include <stdint.h>
#include <gsl/gsl_blas.h> 
#include <gsl/gsl_math.h>
#include "LA_utils.h"
#ifndef NETWORK_H
#define NETWORK_H

class Network { 

public: 
	int num_layers; 
	int* sizes; 
	gsl_vector* biases; 
	gsl_matrix*  weights;
	

	Network (int *node_layers, int num_layers);
 
	gsl_vector* feedforward(gsl_vector* a); 

    void SGD(gsl_vector* training_data, gsl_vector* output_label,
			uint32_t samples, uint32_t batch_size, float learning_rate); 
        /**
		 * Train the neural network using mini-batch stochastic
         * gradient descent.  The ``training_data`` is a list of tuples
         * ``(x, y)`` representing the training inputs and the desired
         * outputs.  The other non-optional parameters are
         * self-explanatory.  If ``test_data`` is provided then the
         * network will be evaluated against the test data after each
         * epoch, and partial progress printed out.  This is useful for
         * tracking progress, but slows things down substantially."""
		 */
 
	void single_batch_iteration(gsl_vector* training_data, gsl_vector* output_label,
			uint32_t batch_size, float l_rate);
        /* Update the network's weights and biases by applying
		 * gradient descent using backpropagation to a single mini batch.
		 * The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
         * is the learning rate."""
		 */
      
    void backprop(gsl_vector* x, gsl_vector* y, gsl_vector* nabla_b, gsl_matrix* nabla_w); 
        /**
		 * Return a tuple ``(nabla_b, nabla_w)`` representing the
         * gradient for the cost function C_x.  ``nabla_b`` and
         * ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
         * to ``self.biases`` and ``self.weights``."""
		 */
     

    uint32_t evaluate(gsl_vector* test_input, int count, gsl_vector* expected_output);
        /**
		 * Return the number of test inputs for which the neural
         * network outputs the correct result. Note that the neural
         * network's output is assumed to be the index of whichever
         * neuron in the final layer has the highest activation.
		 */ 

    gsl_vector* cost_derivative(gsl_vector* output_activations, gsl_vector* y); 
        /**
		 * Return the vector of partial derivatives \partial C_x /
         * \partial a for the output activations."""
		 */

	gsl_vector* sigmoid(gsl_vector* z); 
	    /*
		 * The sigmoid function for activation values for neurons. 
		 */ 
						// expm1 is exp(z) -1. Must readd 1.
	
	gsl_vector* sigmoid_prime(gsl_vector* z); 
	    /**
		 * Derivative of the sigmoid function.
		 */ 
}; 

#endif
