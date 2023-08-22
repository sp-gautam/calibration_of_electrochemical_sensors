import numpy as np
import pickle as pkl
import sklearn 

# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
def my_predict( df ):
    
    models = [1,2]
	# Load your model file
    with open("final_model_o3.pkl", "rb") as f:
        models[0] = pkl.load(f)
    with open("final_model_no2.pkl", "rb") as f:
        models[1] = pkl.load(f)
	# Make two sets of predictions, one for O3 and another for NO2
    pred1 = models[0].predict( df.drop( [ "Time" ], axis = "columns" ).to_numpy() )
    pred2 = models[1].predict( df.drop( [ "Time" ], axis = "columns" ).to_numpy() )
	# Return both sets of predictions
    return ( pred1, pred2 )