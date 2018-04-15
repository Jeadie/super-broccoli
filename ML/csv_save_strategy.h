#include "linear_regression.h"
#include "abstract_save_strategy.h"
#ifndef csv_save_strategy_h
#define csv_save_strategy_h


class CSVSaveStrategy : public AbstractSaveStrategy {

public: 
	
	/**
	 * Represents
	 * Returns: 0 if the saving was successful, else an integer representative of the cause. 
	 */ 


	/**
	 * Loads the Linear Regression model from persistent storage. 
	 */ 
	LinearRegression* load_linear_regression(void* load_details); 

	/**
	 * Saves the Linear Regression model to storage with the given save_details. 
	 */ 
	int save_linear_regression(LinearRegression* lr, void* save_details); 

}; 


#endif 
