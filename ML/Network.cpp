#include <stdio.h> 
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <iostream> 
#include <string.h>
#include <stdint.h>
#include <gsl/gsl_blas.h> 
#include <gsl/gsl_math.h>
#include "LA_utils.h"
#include "Network.h"


	Network::Network (int *node_layers, int num_layers) {
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
	}
 
	gsl_vector* Network::feedforward(gsl_vector* a){
        /**
		 * Return the output of the network if image a is input.
		 */
		if(a->size != (uint8_t) *(this->sizes)) {
			fprintf(stdout, "Input vector is not the same length as input to network.\n"); 
			return (gsl_vector*) 0; 
		}
		gsl_vector_fprintf(stdout, a, "%.5g"); 
		gsl_vector* output = gsl_vector_alloc(a->size); 
		gsl_vector_memcpy(output, a); 
		for (uint8_t i = 0; i < (this->num_layers - 1); i++) {
			gsl_vector* bias = &this->biases[i];
			gsl_matrix* weights = &this->weights[i]; 
			output = matrix_vector_dot(output, weights);
			gsl_vector_add(output, bias); 
			gsl_vector* new_output = sigmoid(output); 
			output = new_output; 
			//swap_and_free(new_output, output); 
		}
		gsl_vector_fprintf(stdout, output, "%.5g"); 
		return output; 
	}

    void Network::SGD(gsl_vector* training_data, gsl_vector* output_label,
			uint32_t samples, uint32_t batch_size, float learning_rate) {
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
		uint32_t batch_count = samples/ batch_size; 
		for (uint32_t i = 0; i < batch_count; i++) {
			single_batch_iteration(&training_data[i * batch_size], &output_label[i * batch_size], batch_size,  learning_rate); 		
		}
	}

    void Network::single_batch_iteration(gsl_vector* training_data, gsl_vector* output_label,
			uint32_t batch_size, float l_rate) {
        /* Update the network's weights and biases by applying
		 * gradient descent using backpropagation to a single mini batch.
		 * The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
         * is the learning rate."""
		 */
        gsl_vector* nabla_b = (gsl_vector*) malloc(sizeof(gsl_vector*) * (this->num_layers - 1)); 
		gsl_matrix* nabla_w = (gsl_matrix*) malloc(sizeof(gsl_matrix*) * (this->num_layers - 1)); 
		
		for (int i = 0; i < this->num_layers - 1; i++) {
		    nabla_b[i] = *gsl_vector_calloc(*(this->sizes + i + 1)); 
			nabla_w[i] = *gsl_matrix_calloc(*(this->sizes+i+1), *(this->sizes+i));
		}
        
		for (uint32_t i = 0; i < batch_size; i++) {
			backprop(&training_data[i], &output_label[i], nabla_b, nabla_w);
		}
		
		for (int i = 0; i < this->num_layers - 1; i++) {
			gsl_matrix_scale(&nabla_w[i], (l_rate/batch_size));
			gsl_matrix_sub(&this->weights[i], &nabla_w[i]); 

			gsl_vector_scale(&nabla_b[i], (l_rate/batch_size));
			gsl_vector_sub(&this->biases[i], &nabla_b[i]); 
		}
	}

    void Network::backprop(gsl_vector* x, gsl_vector* y, gsl_vector* nabla_b, gsl_matrix* nabla_w) {
        /**
		 * Alters nabla_b, nabla_w (representing the gradient for the cost function C) for x.
		 * nabla_b and nabla_w are the incremental changes in self.biases and self.weights. This 
		 * backprop method represents backpropogating a single training data set to calculate its optimal
		 * gradient vector for the batch of gradient descent. 
		 */
		
		// initiate delta_nabla_b
        gsl_vector delta_nabla_b[this->num_layers - 1]; 
		gsl_matrix delta_nabla_w[this->num_layers - 1]; 
		
		for (int i = 0; i < this->num_layers - 1; i++) {
		    delta_nabla_b[i] = *gsl_vector_calloc(*(this->sizes + i + 1)); 
			delta_nabla_w[i] = *gsl_matrix_calloc(*(this->sizes+i+1), *(this->sizes+i));
		}

		// Keep track of feedforward input x to calculate output layer differences
        gsl_vector* activation = x; 
        gsl_vector* activations = (gsl_vector*) malloc(sizeof(gsl_vector*) * (this->num_layers - 1)); 
        gsl_vector* zs =  (gsl_vector*) malloc(sizeof(gsl_vector*) * (this->num_layers - 1)); //list to store all the sigmoided activations (z vectors), layer by layer
		activations = x; //list to store all the activations, layer by layer
	    
		// Feedforward input and keep track of activations and sigmoided activations 
		for (uint8_t i = 0; i < this->num_layers - 1; i++) {
			// weighted = w*i + b; 
			// TODO: create a function for w*i +b (also used in general feedforward) 
			gsl_vector* weighted_inputs = matrix_vector_dot(activation, &this->weights[i]); 
			gsl_vector_add(weighted_inputs, &this->biases[i]); 
			gsl_vector_fprintf(stdout, weighted_inputs, "%.5g");
			fprintf(stdout, "done one weighted%d|\n", weighted_inputs->size);
			*(zs + i) = *weighted_inputs; 

			///// PROBLEM IS HERE. BEOFRE SIZE WAS 12 GOOD> NOW SIZE IS ONE BAD.
			fprintf(stdout, "%d",  weighted_inputs->size);
			fflush(stdout); 
			activation = sigmoid(weighted_inputs); 
			activations[i] = *activation; 
		}
	
        // apply cost derivative to our output and expected
		fprintf(stderr, "\n\n\n\n\n\n\nsize cost der %d %d\n", y->size, activations[this->num_layers - 2].size); 
        gsl_vector* delta = cost_derivative(&activations[this->num_layers - 2], y); 
		gsl_vector_mul(delta, sigmoid_prime(&zs[this->num_layers -2])); 
        
		delta_nabla_b[this->num_layers -1 - 1] = *delta;
        delta_nabla_w[-1] = *vector_products_matrix(delta, &activations[num_layers - 1 -2]); 
		
		// Backprop through entire system. Add desired changes from this test data into delta_nabla_w/b
		for (int i = 2; i < this->num_layers; i++) {
			gsl_vector* z = &zs[this->num_layers-1 -i]; 
			gsl_vector* sp = sigmoid_prime(z); 
			delta = matrix_vector_dot(delta, transpose( &this->weights[this->num_layers + (1-i)])); 
			gsl_vector_mul(delta, sp); 
			delta_nabla_b[-1] = *delta; 
			delta_nabla_w[-1] = *vector_products_matrix(delta, &activations[this->num_layers -i-1]);
		}


		// Add the backpropped weights and biases to those from the batch already. 
		for(int i = 0; i < this->num_layers-1; i++) {
			gsl_vector_add(&nabla_b[i], &delta_nabla_b[i]); 
			gsl_matrix_add(&nabla_w[i], &delta_nabla_w[i]); 
		}
	}



    uint32_t Network::evaluate(gsl_vector* data, uint32_t test_points, gsl_vector* expected_results) {
        /**Return the number of test inputs for which the neural
         * network outputs the correct result. Note that the neural
         * network's output is assumed to be the index of whichever
         * neuron in the final layer has the highest activation.
		 */
		if(expected_results->size != test_points) {
			return -1; 
		}
		int correctly_evaluated = 0; 
		for(uint32_t i = 0; i < test_points; i++) {
			gsl_vector* end_nodes = feedforward(&data[i]); 
			int max = gsl_vector_max_index(end_nodes); 
			if(max == gsl_vector_get(expected_results, i)) {
				correctly_evaluated++; 
			}
		}
		return correctly_evaluated; 
	}
    
	gsl_vector* Network::cost_derivative(gsl_vector* output_activations, gsl_vector* y) {
        /**
		 * Return the vector of partial derivatives \partial C_x /
         * \partial a for the output activations."""
		 */
        gsl_vector* output = duplicate_vector(output_activations); 
		gsl_vector_sub(output, y);
		return output;
	}

	gsl_vector* Network::sigmoid(gsl_vector* z) { 
	    /*
		 * The sigmoid function for activation values for neurons. 
		 */ 
						// expm1 is exp(z) -1. Must readd 1.
		gsl_vector* new_z  = duplicate_vector(z); 
		for (uint8_t i = 0; i < z->size; i++) {
			double elem = gsl_vector_get(new_z, i); 
			gsl_vector_set(new_z, i, (double) 1.0/( 1.0 + (gsl_expm1(-1.0* elem) + 1.0))); 
		}
		return new_z; 
	}
	
	gsl_vector* Network::sigmoid_prime(gsl_vector* z){
	    /**
		 * Derivative of the sigmoid function.
		 */ 
		gsl_vector* temp_v = duplicate_vector(z); 
		gsl_vector* new_v = sigmoid(temp_v); 
		swap_and_free(new_v, temp_v); 
		temp_v = duplicate_vector(new_v);  

		// return sigmoid(z) / (1-sigmoid(z))
		gsl_vector_scale(temp_v, -1.0); 
		gsl_vector_add_constant(temp_v, 1.0); 
		gsl_vector_div(new_v, temp_v); 
		gsl_vector_free(temp_v); 
		return new_v; 
	}



	/**~Network () 
	{

	}*/
