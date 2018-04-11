#include <stdio.h> 
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "linear_regression.h"
using namespace std;

LinearRegression::LinearRegression() {}
LinearRegression::~LinearRegression() {}
LinearRegression::LinearRegression(gsl_vector* x_vals, gsl_vector* r_vals, int count){
		/**
		 * Sets up the linear regression model from the given supervised data. 
		 * x_vals: input variable such that each vector has equal dimensions. 
		 * r_vals: corresponding outputs for each x_val at corresponding index. 
		 * count: Number of datapoints to build the regression model with.
		 */ 
		this->w1 = gsl_vector_calloc(x_vals->size +1); 
		
		double x; 
		double x_2 = 0; 
		// Construct normal equations: X^tXw = X^tr
		// Construct initial matrix of one extra column for constant term
		gsl_matrix* X = gsl_matrix_calloc(count, x_vals->size + 1); 

		// Fill first column with ones
		gsl_matrix_set_all(X, 1); 

		// Fill matrix X with datapoints for x_vals; 
		for(int i = 0; i < count; i++) {
			gsl_vector_view datapoint = gsl_matrix_subrow(X, i, 1, x_vals->size);
			datapoint.vector.data = x_vals[i].data;
		}


		// Calculate XT x X
		gsl_matrix* X_TX = gsl_matrix_alloc(x_vals->size + 1, x_vals->size +1); 
		gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X, 0.0, X_TX);
		
		// Use a LU decomposition to find the inverse of XT x X
		gsl_matrix *inv_X_TX = gsl_matrix_alloc(x_vals->size +1, x_vals->size +1);  
		int s; 
		gsl_permutation *p = gsl_permutation_alloc(x_vals->size +1); 

		gsl_linalg_LU_decomp(X_TX, p, &s); 
		gsl_linalg_LU_invert(X_TX, p, inv_X_TX); 
	
		// Calculate XT x R
		gsl_vector* X_TR = gsl_vector_alloc(x_vals->size); 
		gsl_blas_dgemv(CblasTrans, 1.0, X, r_vals, 0.0, X_TR); 

		// Calculate weights = inv(XT x X) x (XT x R)  
		gsl_blas_dgemv(CblasNoTrans, 1.0, inv_X_TX, X_TR, 0.0, this->w1); 

		// Memory Cleanup
		gsl_matrix_free(X); 
		gsl_matrix_free(X_TX); 
		gsl_matrix_free(inv_X_TX); 
		gsl_vector_free(X_TR); 
		gsl_permutation_free(p); 
	}

	double LinearRegression::regress(gsl_vector* x) {
		/**
		 * Returns the output value from the linear regression model at point x. 
		 */ 
		gsl_vector* x_with_constant_term = this->construct_standard_linear_input(x); 
		double y = 0; 
		gsl_blas_ddot(this->w1, x_with_constant_term, &y); 
		gsl_vector_free(x_with_constant_term); 
		return y; 
	}
	


	gsl_vector* LinearRegression::construct_standard_linear_input(gsl_vector* inputs) {
		/**
		 * Returns a vector, v such that v[0] = 1 & v[i] = inputs[i-1] for all i > 0. 
		 */
		gsl_vector* constant_termed_vector = gsl_vector_alloc(inputs->size +1); 
		
		// add vector inputs into subvector of newly created vector starting at index 1. 
		gsl_vector_view constant_termed_subvector =  gsl_vector_subvector(constant_termed_vector, 1, inputs->size); 
		constant_termed_subvector.vector.data = inputs->data; 
		return constant_termed_vector; 
	}





