#include <stdio.h> 
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#ifndef LIN_REG
#define LIN_REG
using namespace std;

class LinearRegression {
public: 
	
	/* Destructor */ 
	~LinearRegression();

	/**
	 * Sets up the linear regression model from the given supervised data. 
	 * x_vals: input variable such that each vector has equal dimensions. 
	 * r_vals: corresponding outputs for each x_val at corresponding index. 
	 * count: Number of datapoints to build the regression model with.
	 */ 
	LinearRegression(gsl_vector* x_vals, gsl_vector* r_vals, int count);
	

	/**
	 * Returns the output value from the linear regression model at point x. 
	 */
	double regress(gsl_vector* x); 
	 
		
	/**
	 * Returns the weight vector derived from the regression model. 
	 */ 
	gsl_vector* get_weight_vector(); 

	/**
	 * Sets the model's weights to the given input. 
	 */ 
	void set_weight_vector(gsl_vector* weights); 
	

private: 

	// Linear Gradient vector  
	gsl_vector* weights;

	/**
	 * Returns a vector, v such that v[0] = 1 & v[i] = inputs[i-1] for all i > 0. 
	 */
	gsl_vector* construct_standard_linear_input(gsl_vector* inputs);

};

#endif

