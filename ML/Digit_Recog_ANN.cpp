#include <stdio.h> 
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <iostream> 
#include "data_parser.h"
#include <string.h>
#include <stdint.h>
#include <gsl/gsl_blas.h> 
#include <gsl/gsl_math.h>
#include "Network.h"
#include <stdio.h>

int main() {
	int* layers = (int* ) malloc(sizeof(int) *3); 
	*layers = 5; 
	*(layers + 1) = 12; 
	*(layers + 2) = 3; 
	Network *net = new Network(layers, 3); 
	
	
	return 0;
}
