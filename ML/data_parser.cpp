#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <stdint.h>

typedef struct number_image_pairs {
	// TODO: Convert to gsl_matrix and int
	gsl_matrix_uchar *images; 
	gsl_vector_uchar *labels;
	int count; 
} number_image_pairs;

int32_t bytes_to_int32_t (char *bytes) {
	return ((int32_t) *(bytes + 3)) +
				((int32_t)*(bytes + 2) << 8) +
				((int32_t)*(bytes + 1) << 16) +
				((int32_t)*(bytes + 0) << 24);
}

uint32_t bytes_to_uint32_t(char *bytes) { 
	return ((uint32_t) *bytes) +
				((uint32_t)*(bytes + 1) << 8) +
				((uint32_t)*(bytes + 2) << 16) +
				((uint32_t)*(bytes + 3) << 24);
}

gsl_matrix_uchar* Get_Number_Images(FILE *file) {
	char m_number[4], b_count[4], b_rows[4], b_columns[4];

	if(fread(m_number, 1, 4, file) != 4||
			fread(b_count, 1, 4, file) != 4 ||
			fread(b_rows, 1, 4, file) != 4||
			fread(b_columns, 1, 4, file) !=4){ 
		fprintf(stdout, "Error reading header."); 
		return (gsl_matrix_uchar * ) 0; 
	}
	if(m_number[0] != 0 || m_number[1] != 0) { 
		fprintf(stdout, "Invalid Header."); 
		return (gsl_matrix_uchar * ) 0; 
	}
	
	int32_t rows = bytes_to_int32_t (b_rows); 
	int32_t count = 60000; //bytes_to_int32_t (b_count); 
	int32_t columns = bytes_to_int32_t (b_columns); 
	fprintf(stdout, "rows %d\n columns %d\n count %d\n", rows, columns, count); 
	gsl_matrix_uchar * images; 
	for (int32_t i = 0; i < count; i++) {
		gsl_matrix_uchar* image = gsl_matrix_uchar_calloc(rows, columns); 
		if (gsl_matrix_uchar_fread(file, image) != 0) {
			fprintf(stdout, "Error reading image from file."); 
			return (gsl_matrix_uchar* ) 0; 
		}; 
		fprintf(stdout, "%d\n", i); // gsl_matrix_uchar_fprintf(stdout, image, "%d"); 
		images[i] = *image; 
	}
	return images; 
};

gsl_vector_uchar* Get_Number_Labels(FILE *file) {
	char label_header[8]; 
	
	int read = fread(label_header, 1, 8, file); 
	if( read != 8) {
		fprintf(stdout, "Can't read header bytes from labels: %d.", read); 
		return (gsl_vector_uchar*) 0; 
	}
	if( label_header[0] || label_header[1]) {
		fprintf(stdout, "Header is incorrectly Configured."); 
		return (gsl_vector_uchar*) 0; 
	}
	char byte_type = label_header[2]; 
	int vector_dimension = label_header[3]; 
	//TODO: fix to parsing the 4 bytes 
	int32_t  label_count = 60000; //((int32_t) label_header[4] << 24) |  ((int32_t) label_header[5] << 16) |  ((int32_t) label_header[6] << 8) |  ((int32_t) label_header[7]); 

//	fprintf(stdout, "count: %u %u %u %u %u %u %u %u",  label_header[6], label_header[1], label_header[2], label_header[3], label_header[4], label_header[5], label_header[6], label_header[7]);
	fflush(stdout); 
	gsl_vector_uchar* labels = gsl_vector_uchar_alloc(label_count); 
	for (int32_t i = 0; i < label_count; i++) {
		unsigned char byte = fgetc(file); 
		//fprintf(stdout, "%c\n", byte); 
		gsl_vector_uchar_set(labels, i, byte);  
	}
	return labels; 
}; 

void number_image_pairs_free(number_image_pairs *pairs) {
	gsl_vector_uchar_free(pairs->labels); 
	for (int i =0; i < pairs->count; i++) {
		gsl_matrix_uchar_free(&(pairs->images[i])); 
	}
	free(pairs); 
}
number_image_pairs* Get_Number_Images_From_Idx(char* image_path, char* label_path) {
	FILE* image_file = fopen(image_path, "r"); 
	FILE* label_file = fopen(label_path, "r");	
	if( image_file == 0 || label_file ==0) {
		fprintf(stdout, "Failed to open image or label files"); 
		return (number_image_pairs*) 0; 
	}
	number_image_pairs *pairs = (number_image_pairs*) malloc(sizeof(number_image_pairs));	 
	pairs->images = Get_Number_Images(image_file); 
	pairs->labels = Get_Number_Labels(label_file); 
	pairs->count = 10; //pairs->labels->size;		
	//gsl_vector_uchar_fprintf(stdout, pairs->labels, "%d"); 
	if(pairs->images == 0 || pairs->labels == 0) {
		//TODO: Properly free non complete inner data structures
		return (number_image_pairs *) 0;
	}
	fclose(image_file); 
	fclose(label_file); 
	return pairs; 
};


