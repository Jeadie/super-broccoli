#include <stdio.h> 
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <iostream> 
#include "data_parser.h"
#include <string.h>
#include <stdint.h>
#include <gsl/gsl_blas.h> 
#include <gsl/gsl_math.h>

class Network { 

	public: 
	int num_layers; 
	int* sizes; 
	gsl_vector* biases; 
	gsl_matrix*  weights;
	

	Network (int *node_layers, int num_layers) {
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
 
	gsl_vector* feedforward(gsl_vector* a){
        /**
		 * Return the output of the network if image a is input.
		 */
		if(a->size != (uint8_t) *(this->sizes)) {
			fprintf(stdout, "Input vector is not the same length as input to network.\n"); 
			return (gsl_vector*) 0; 
		}
		gsl_vector* output = gsl_vector_alloc(a->size); 
		gsl_vector_memcpy(output, a); 
		for (uint8_t i = 0; i < *this->sizes; i++) {
			gsl_vector* bias = &this->biases[i];
			gsl_matrix* weights = &this->weights[i]; 
			matrix_vector_dot(output, weights); 
			gsl_vector_add(output, bias); 
			gsl_vector* new_output = sigmoid(output); 
			swap_and_free(new_output, output); 
		}
		return output; 
	}
	
	gsl_vector* matrix_vector_dot(gsl_vector* a, gsl_matrix* w) {
		gsl_vector* temp_vect = gsl_vector_alloc(w->size2); 
		
		for(uint8_t i = 0; i < w->size2; i++ ) {
			double* dot_value; 
			gsl_vector x = gsl_matrix_column(w, i).vector;  
			gsl_blas_ddot(a, &x, dot_value); 
			gsl_vector_set(temp_vect, i, *dot_value);
		}
		
		return temp_vect; 
	}

    void SGD(gsl_vector* training_data, gsl_vector* output_label,
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

    void single_batch_iteration(gsl_vector* training_data, gsl_vector* output_label,
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

    void backprop(gsl_vector* x, gsl_vector* y, gsl_vector* nabla_b, gsl_matrix* nabla_w) {
        /**
		 * Return a tuple ``(nabla_b, nabla_w)`` representing the
         * gradient for the cost function C_x.  ``nabla_b`` and
         * ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
         * to ``self.biases`` and ``self.weights``."""
		 */
        gsl_vector delta_nabla_b[this->num_layers - 1]; 
		gsl_matrix delta_nabla_w[this->num_layers - 1]; 
		
		for (int i = 0; i < this->num_layers - 1; i++) {
		    delta_nabla_b[i] = *gsl_vector_calloc(*(this->sizes + i + 1)); 
			delta_nabla_w[i] = *gsl_matrix_calloc(*(this->sizes+i+1), *(this->sizes+i));
		}

		// feedforward
        gsl_vector* activation = x; 
        gsl_vector activations[this->num_layers - 1]; 
		activations[0] = *x; //list to store all the activations, layer by layer

        gsl_vector zs[this->num_layers - 1]; //list to store all the z vectors, layer by layer

	    for (int i = 0; i < this->num_layers - 1; i++) {
			// weighted = w*i + b; 
			gsl_vector* weighted_inputs = matrix_vector_dot(activation, &this->weights[i]); 
			gsl_vector_add(weighted_inputs, &this->biases[i]); 
			zs[i] = *weighted_inputs; 	// TODO: switch matrix_vector_dot to new return result. 
			activation = sigmoid(weighted_inputs); 
			activations[i] = *activation; 
		}

        // backward pass
        gsl_vector* delta = cost_derivative(&activations[this->num_layers - 2], y); 
		gsl_vector_mul(delta, sigmoid_prime(&zs[this->num_layers -2])); 
        
		delta_nabla_b[this->num_layers -1 - 1] = *delta;
        delta_nabla_w[-1] = *vector_products_matrix(delta, &activations[num_layers - 1 -2]); 
		
		
		/** Note that the variable l in the loop below is used a little
         * differently to the notation in Chapter 2 of the book.  Here,
         *  l = 1 means the last layer of neurons, l = 2 is the
         * second-last layer, and so on.  It's a renumbering of the
         * scheme in the book, used here to take advantage of the fact
         * that Python can use negative indices in lists.
		 */ 
		for (int i = 2; i < this->num_layers; i++) {
			gsl_vector* z = &zs[this->num_layers-1 -i]; 
			gsl_vector* sp = sigmoid_prime(z); 
			delta = matrix_vector_dot(delta, transpose( &this->weights[this->num_layers + (1-i)])); 
			gsl_vector_mul(delta, sp); 
			delta_nabla_b[-1] = *delta; 
			delta_nabla_w[-1] = *vector_products_matrix(delta, &activations[this->num_layers -i-1]);
		}

        /**for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w) */ 
		for(int i = 0; i < this->num_layers-1; i++) {
			gsl_vector_add(&nabla_b[i], &delta_nabla_b[i]); 
			gsl_matrix_add(&nabla_w[i], &delta_nabla_w[i]); 
		}
	}

	gsl_matrix* transpose(gsl_matrix* m) {
		gsl_matrix* transposed = gsl_matrix_calloc(m->size2, m->size1); 
	
		for(uint8_t i = 0; i < m->size1; i++) {
			gsl_vector* transfer_vector = gsl_vector_calloc(m->size2); 
			gsl_matrix_get_row(transfer_vector, m, i); 
			gsl_matrix_set_row(transposed, i, transfer_vector); 
			gsl_vector_free(transfer_vector); 
		}
		return transposed; 

	}
	
	gsl_matrix* vector_products_matrix(gsl_vector* v1, gsl_vector* v2) {
		/**
		 * Returns a matrix (v1->size, v2->size) matrix such that m(i, j) = v1[i] * v2[j]
		 */ 
		gsl_matrix* product_m = gsl_matrix_calloc(v1->size, v2->size);
		
		// Iterate over each column, set it to v1 then multiply by v2[i]
		for(uint8_t i = 0; i < v2->size; i++) {
			gsl_vector matrix_column = gsl_matrix_column(product_m, i).vector;
			gsl_vector_memcpy(&matrix_column, v1); 
			gsl_vector_scale(&matrix_column, gsl_vector_get(v2, i)); 
		}
		return product_m; 
	}


    /**def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

	*/ 
    gsl_vector* cost_derivative(gsl_vector* output_activations, gsl_vector* y) {
        /**
		 * Return the vector of partial derivatives \partial C_x /
         * \partial a for the output activations."""
		 */
        gsl_vector* output = duplicate_vector(output_activations); 
		gsl_vector_sub(output, y);
		return output;
	}

	void swap_and_free(gsl_vector* src, gsl_vector* dest) {
		/**
		 * Points the src vector to destination and frees the original dest vector.  
		 */ 
		
		gsl_vector* temp = dest; 
		dest = src; 
		gsl_vector_free(temp); 
	}

	gsl_vector* sigmoid(gsl_vector* z) { 
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
	
	gsl_vector* sigmoid_prime(gsl_vector* z){
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

	gsl_vector* duplicate_vector(gsl_vector* v) {
		gsl_vector* duplicate = gsl_vector_alloc(v->size); 
		gsl_vector_memcpy(duplicate, v);
		return duplicate; 
	}



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
	gsl_matrix_uchar_fprintf(stdout, &pairs->images[22], "%d");  
	printf("count: %d\n", pairs->count); 

}

