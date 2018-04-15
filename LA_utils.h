#include <gsl/gsl_matrix.h>

gsl_vector* matrix_vector_dot(gsl_vector* a, gsl_matrix* w); 

gsl_matrix* transpose(gsl_matrix* m);
	
gsl_matrix* vector_products_matrix(gsl_vector* v1, gsl_vector* v2);

void swap_and_free(gsl_vector* src, gsl_vector* dest); 

gsl_vector* duplicate_vector(gsl_vector* v); 

