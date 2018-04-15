#include <stdio.h>
#include "linear_regression.h"
#include "csv_save_strategy.h" 
#include "abstract_save_strategy.h"

using namespace std; 

LinearRegression* CSVSaveStrategy::load_linear_regression(void* load_details) { 
	char* filepath = (char*) load_details; 
	
	

}



int CSVSaveStrategy::save_linear_regression(LinearRegression* lr, void* save_details) {
	char* loadpath = (char* ) save_details; 
	
	


	return 0; 

}





