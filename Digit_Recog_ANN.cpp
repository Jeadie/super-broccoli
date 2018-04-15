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
#include <gsl/gsl_math.h>
#include "linear_regression.h"


int main() {
	
	gsl_vector* a = gsl_vector_alloc(1); 
	gsl_vector* b = gsl_vector_alloc(1); 
	gsl_vector* c = gsl_vector_alloc(1); 
	gsl_vector x_vals[3]; 
	x_vals[0]= *a; 
	x_vals[1] = *b; 
	x_vals[2] = *c; 
	gsl_vector_set(a, 0, -1); 
	gsl_vector_set(b, 0, 1); 
	gsl_vector_set(c, 0, 2); 		

	gsl_vector* y_vals = gsl_vector_alloc(3); 
	gsl_vector_set(y_vals, 0, 0);
	gsl_vector_set(y_vals, 1, 2); 
	gsl_vector_set(y_vals, 2, 3); 

	LinearRegression lr = LinearRegression(x_vals, y_vals, 3); 
	fprintf(stdout, "%f, %f, %f, %f", lr.regress(gsl_vector_alloc(1)), lr.regress(b), lr.regress(c)); //, lr->regress(2.5)); 
	return 0; 

	int* layers = (int* ) malloc(sizeof(int) *3); 
	*layers = 5; 
	*(layers + 1) = 12; 
	*(layers + 2) = 3; 
	Network *net = new Network(layers, 3); 
	
	gsl_vector* output = gsl_vector_calloc(10); 
	gsl_vector inputs[10]; 
	for(int i = 0; i < 10; i++) {
		gsl_vector* value = gsl_vector_calloc(5); 
		for(int j =0; j < 5; j++) {
			gsl_vector_set(value, j, gsl_log1p(j*i)); 
		}
		inputs[i] = *value; 
		gsl_vector_set(output, i, i/4); 
	}
	
	net->SGD(inputs, output, 10, 1, 0.1);  
	int correct = net->evaluate(inputs, 10, output); 
	fprintf(stdout, "correct: %d", correct);
	
	
	return 0;
}
