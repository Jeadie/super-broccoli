#include <gsl/gsl_matrix.h>

typedef struct number_image_pairs {
	// TODO: Convert to gsl_matrix and int
	gsl_matrix_uchar *images; 
	gsl_vector_uchar *labels;
	int count; 
} number_image_pairs;


void number_image_pairs_free(number_image_pairs *pairs); 

number_image_pairs* Get_Number_Images_From_Idx(char* image_path, char* label_path); 




