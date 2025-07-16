import uuid 
import os 
import sys 
import json 
import glob 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pymc as pm
import arviz as az
az.style.use('dark_background')
plt.style.use('dark_background')


def my_test(): 

    # =============================
    # TESTING SIMPLE MODEL 
    
    files = glob.glob(os.path.join(os.path.dirname(__file__), "*.csv"))
    
    simple_model_0(files, score_col= "proposed_tech_score")


    return 

def simple_model_0(data_sources, score_col = "score", display = True): 

    print(data_sources)
    # merge data_sources 
    dataframes = [pd.read_csv(file) for file in data_sources]
    
    # Concatenate all DataFrames into one
    merged_data_source = pd.concat(dataframes, ignore_index=True).sample(frac = 1).reset_index(drop = True)
    

    entities_idx, entities = pd.factorize(merged_data_source['from'], sort = True) 
    coords = {"entities": entities}

    print(coords)
    

    with pm.Model(coords =coords) as competitor_model: 
    
        mu = pm.HalfNormal("mu", sigma = 10, dims ="entities")
        alpha = pm.HalfNormal("alpha", sigma = 10)

        beta = alpha / mu[entities_idx]
        score = pm.Gamma("score", 
                        alpha= alpha, 
                        beta = beta, 
                        observed=merged_data_source[score_col].values)
        
        trace = pm.sample(1000, tune = 1000)
    
    az.plot_forest(trace, var_names=["mu"], combined=True)
    plt.title("Posterior means for each competitor")
    if display: 
        plt.show()
    else: 
        fname = "/tmp/{}.png".format(str(uuid.uuid4()))
        plt.savefig(fname)
        return fname 