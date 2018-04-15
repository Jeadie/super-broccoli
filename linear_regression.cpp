#include <stdio.h> 
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "linear_regression.h"
using namespace std;

LinearRegression::~LinearRegression() {
	gsl_vector_free(this->get_weight_vector()); 
}

LinearRegression::LinearRegression(gsl_vector* x_vals, gsl_vector* r_vals, int count){
		// init linear weights vector (constant term and each feature)
		this->set_weight_vector(gsl_vector_calloc(x_vals->size +1)); 
		
		// Construct normal equations: X^tXw = X^tr
		// Construct initial matrix of one extra column for constant term with enough rows for each datapoint. 
		gsl_matrix* X = gsl_matrix_calloc(count, x_vals->size + 1); 

		// Fill matrix X with datapoints for x_vals; 
		for(int i = 0; i < count; i++) {
			gsl_vector* row = construct_standard_linear_input(&x_vals[i]); 
			gsl_matrix_set_row(X, i, row); 
		}
		
		// Calculate XT x X
		gsl_matrix* X_TX = gsl_matrix_alloc(x_vals->size + 1, x_vals->size +1); 
		gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X, 0.0, X_TX);
		
		// Use a LU decomposition to find the inverse of XT x X
		gsl_matrix *inv_X_TX = gsl_matrix_calloc(x_vals->size +1, x_vals->size +1);  
		int s ;  
		gsl_permutation *p = gsl_permutation_alloc(x_vals->size +1); 
		
		gsl_linalg_LU_decomp(X_TX, p, &s);
		gsl_linalg_LU_invert(X_TX, p, inv_X_TX); 
	
		// Calculate XT x R
		gsl_vector* X_TR = gsl_vector_alloc(x_vals->size + 1); 
		gsl_blas_dgemv(CblasTrans, 1.0, X, r_vals, 0.0, X_TR); 

		// Calculate weights = inv(XT x X) x (XT x R)  
		gsl_blas_dgemv(CblasNoTrans, 1.0, inv_X_TX, X_TR, 0.0, this->get_weight_vector());

		// Memory Cleanup
		gsl_matrix_free(X); 
		gsl_matrix_free(X_TX); 
		gsl_matrix_free(inv_X_TX); 
		gsl_vector_free(X_TR); 
		gsl_permutation_free(p); 
	}

	double LinearRegression::regress(gsl_vector* x) {
		gsl_vector* x_with_constant_term = this->construct_standard_linear_input(x); 
		double y = 0; 
		gsl_blas_ddot(this->get_weight_vector(), x_with_constant_term, &y); 
		gsl_vector_free(x_with_constant_term); 
		return y; 
	}
	
	gsl_vector* LinearRegression::get_weight_vector(){
		return this->weights; 
	}

	void LinearRegression::set_weight_vector(gsl_vector* weights) {
		this->weights = weights; 
	}


	gsl_vector* LinearRegression::construct_standard_linear_input(gsl_vector* inputs) {
		// Create extended vector to account for constant term. 
		gsl_vector* constant_termed_vector = gsl_vector_alloc(inputs->size +1); 
		
		// add input into subvector of newly created vector starting at index 1. 
		gsl_vector_view constant_termed_subvector =  gsl_vector_subvector(constant_termed_vector, 1, inputs->size); 
		gsl_vector_memcpy(&constant_termed_subvector.vector, inputs); 
		gsl_vector_set(constant_termed_vector, 0, 1); 

		return constant_termed_vector; 
	}





