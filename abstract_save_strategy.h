#include "linear_regression.h"
#ifndef abstract_save_strategy_h
#define abstract_save_strategy_h


class AbstractSaveStrategy {

public: 
	
	/**
	 *	Adding Machine Learning models to the strategy requires two function prototypes: 
	 *		Model* load_model(void* load_details); 
	 *			where load_details refers to details required for the load strategy. 
	 *
	 *		int save_model(Model* model, void* save_details); 
	 *			Where save_details refers to details required for the save strategy and Model is the 
	 *			data structure to be saved. NOTE: there is no explicit Model class. 
	 *			Returns: 0 if the saving was successful, else an integer representative of the cause. 
	 */ 


	/**
	 * Loads the Linear Regression model from persistent storage. 
	 */ 
	virtual LinearRegression* load_linear_regression(void* load_details); 

	/**
	 * Saves the Linear Regression model to storage with the given save_details. 
	 */ 
	virtual int save_linear_regression(LinearRegression* lr, void* save_details); 

}; 


#endif 
