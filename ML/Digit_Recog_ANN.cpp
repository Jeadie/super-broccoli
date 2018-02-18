#include <stdio.h> 
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <iostream> 

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
	net->print_biases(); 
		
}

