#include <gsl/gsl_matrix.h>
#include <stdint.h>
#include <gsl/gsl_blas.h> 

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


void swap_and_free(gsl_vector* src, gsl_vector* dest) {
	/**
	 * Points the src vector to destination and frees the original dest vector.  
	 */ 
		
	gsl_vector* temp = dest; 
	dest = src; 
	gsl_vector_free(temp); 
}

gsl_vector* duplicate_vector(gsl_vector* v) {
	gsl_vector* duplicate = gsl_vector_alloc(v->size); 
	gsl_vector_memcpy(duplicate, v);
	return duplicate; 
}

