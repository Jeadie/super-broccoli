#include "linear_regression.h"
#include "abstract_save_strategy.h"

using namespace std; 
	
	/**
	 * Base Save strategy functions for the ML models. 
	 *
	 */ 

	LinearRegression* AbstractSaveStrategy::load_linear_regression(void* load_details) {
		return (LinearRegression*) 0; 
	}

	int AbstractSaveStrategy::save_linear_regression(LinearRegression* lr, void* save_details) {
		return 0; 
	}


