#include <stdio.h> 
#include <gsl/matrix/gsl_matrix.h>
#include <gsl/rng/gsl_rng.h> 
#include <iostream> 

class Network { 

	protected: 
	int num_layers; 
	int* sizes; 
	gsl_vector* biases; 
	gsl_matrix*  weights;
	

	public:
	Network (int *node_layers, int num_layers) 
	{
	num_layers = num_layers; 
	sizes = node_layers; 
	biases = (gsl_vector*) malloc((num_layers - 1) * sizeof(gsl_vector));
	weights = (gsl_matrix*) malloc((num_layers - 1) * sizeof(gsl_matrix)); 
 // self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
 //        self.weights = [np.random.randn(y, x)
 //                       for x, y in zip(sizes[:-1], sizes[1:])]

	gsl_rng * rng = gsl_rng_alloc(gsl_rng_rand); 
	gsl_rng_set(rng, 8080); 
	// Instantiate bias vectors
	for (int i = 1; i < num_layers; i++) {
		int nodes = *(sizes + i); 
		gsl_vector *bias_layer = gsl_vector_alloc(nodes); 
		for (int j = 0; i < nodes; i++) {
			gsl_vector_set(bias_layer, j, gsl_rng_uniform_pos(rng)); 
		}
		*(biases + i) = *bias_layer;
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
			*(weights + i) = *weight_layer; 
		}
	}
	gsl_rng_free(rng); 
	}

	~Network () 
	{

	}
	void print_biases(); 

};

void Network:: print_biases(){

	std::cout << num_layers; 

};

int main(int argc, char *argv[]) {
	int layers[3] = {5, 10, 1}; 

	Network net = Network(layers, 3); 
	net.print_biases(); 
		
}

