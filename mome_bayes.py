import inspect 
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
    
    # files = glob.glob(os.path.join(os.path.dirname(__file__), "*.csv"))
    # files = glob.glob(os.path.join(os.path.expanduser("~"), "tmp_data", "*.csv"))
    # simple_model_0(files, score_col= "score")

    files= glob.glob(os.path.join('/tmp', "*_subset.csv"))
    print(simple_binary_competition(files, score_col="pilled", coords_name="ceo", display=True))


    return 

def save_model_plots(trace, entities, model_graph_path):
    """
    Generate and save diagnostic plots for the model.

    Args:
        trace: The ArviZ InferenceData object.
        entities: The list/array of entity names.
        model_graph_path: Path to the saved model graph image.

    Returns:
        dict: Mapping of plot names to their file paths.
    """
    image_paths = {}

    folder = os.path.dirname(model_graph_path)
    
    # Forest plot
    plt.figure()
    az.plot_forest(trace, var_names=["mu"], combined=True)
    forest_path = os.path.join(folder, 'forest_plot.png')
    plt.savefig(forest_path, bbox_inches='tight')
    image_paths['forest_plot'] = forest_path
    plt.close()

    # Posterior plot
    plt.figure()
    az.plot_posterior(trace, var_names=["mu"], coords={"entities": entities}, hdi_prob=0.95, figsize=(8, len(entities)*1.5))
    posterior_path = os.path.join(folder, 'posterior_plot.png')
    plt.savefig(posterior_path, bbox_inches='tight')
    image_paths['posterior_plot'] = posterior_path
    plt.close()

    # Trace plot
    plt.figure()
    az.plot_trace(trace, var_names=["mu", "alpha"])
    trace_path = os.path.join(folder, 'trace_plot.png')
    plt.savefig(trace_path, bbox_inches='tight')
    image_paths['trace_plot'] = trace_path
    plt.close()

    # Add model graph
    image_paths['model_graph'] = model_graph_path

    return image_paths



def build_simple_model_0(data, coords, entities_idx, target_col = 'score'): 

    with pm.Model(coords=coords) as model:
        mu = pm.HalfNormal("mu", sigma=10, dims="entities")
        alpha = pm.HalfNormal("alpha", sigma=10)
        beta = alpha / mu[entities_idx]
        score = pm.Gamma("score", alpha=alpha, beta=beta, observed=data[target_col])
    return model


def render_graph_model(graph_model, save_folder, fname ="model_graph.png"): 
    with graph_model: 
        graph = pm.model_to_graphviz(graph_model)

    graph_path = os.path.join(save_folder, fname)
    graph.render(graph_path[:-4], format = "png", cleanup=True)
    return graph_path 

def simple_model_0(data_sources, score_col = "score", display = True, save_folder = "/tmp"): 

    # merge data_sources 
    dataframes = [pd.read_csv(file) for file in data_sources]
    
    # Concatenate all DataFrames into one
    merged_data_source = pd.concat(dataframes, ignore_index=True).sample(frac = 1).reset_index(drop = True)

    entities_idx, entities = pd.factorize(merged_data_source['from'], sort = True) 
    coords = {"entities": entities}
    
    competitor_model = build_simple_model_0(merged_data_source, coords, entities_idx, score_col)
    
    graph_path = render_graph_model(competitor_model, save_folder)

    with competitor_model:     
        trace = pm.sample(1000, tune = 1000)

    trace_path = os.path.join(os.path.dirname(graph_path), "trace.nc")
    az.to_netcdf(trace, trace_path)
    image_paths = save_model_plots(trace, entities, graph_path)

    # Optionally display plots
    if display:
        for img in image_paths.values():
            img_data = plt.imread(img)
            plt.figure()
            plt.imshow(img_data)
            plt.axis('off')
        plt.show()

    # Build resulting dict
    resulting_dict = {
        "trace": trace_path,
        "image_paths": image_paths,
        "model_builder_file": __file__, 
        "model_builder_function": inspect.currentframe().f_code.co_name,
    }

    # print(json.dumps(resulting_dict, indent = 4))

    return resulting_dict

def build_model_simple_binary_competition(agg, coords, score_col): 


    with pm.Model(coords = coords) as model:
        alpha = pm.Normal("alpha", 0.,1. , dims = 'ceo')
        prob  =pm.Deterministic(f"p_{score_col}", pm.math.sigmoid(alpha))
        k_obs = pm.Binomial("k_obs", n=agg['n'].values, p=prob, observed=agg['successes'].values)

    return model 

def simple_binary_competition_plot(trace, agg, graph_path, score_col, entities_col): 

    image_paths = {}
    
    folder = os.path.dirname(graph_path)

    entities = agg[entities_col].unique()

    da = trace.posterior[f'p_{score_col}']
    p_mean = da.mean(dim = ["chain", "draw"])
    p_lo = da.quantile(0.05, dim = ["chain", "draw"])
    p_hi = da.quantile(0.95, dim = ["chain", "draw"])
    entities_ids = agg[entities_col].values
    obs_rate = agg['successes'] / agg['n']
    yerr = np.vstack([p_mean - p_lo, p_hi - p_mean])
    plt.scatter(entities_ids, obs_rate, color="white", label="Observed proportion", zorder=3)
    plt.errorbar(
        entities_ids, p_mean, yerr=yerr,
        fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=3,
        label="Posterior mean ±90% CI"
    )
    plt.xlabel(f"{entities_col.upper()} (identifier)")
    plt.ylabel(f"P({score_col})")
    plt.xticks(rotation=45)
    plt.ylim(-0.02, 1.02)
    plt.title(f"Per-{entities_col} Binomial model")
    plt.grid(alpha=0.2)
    plt.legend()
    trace_path = os.path.join(folder, "binomial_model_{}.jpg".format(entities_col.replace('_', ' ').title()))
    plt.savefig(trace_path, bbox_inches='tight')
    image_paths['trace_plot'] = trace_path
    plt.close()

    
    # Posterior plot
    plt.figure()
    az.plot_trace(trace, var_names=[f'p_{score_col}'])
    posterior_path = os.path.join(folder, 'simple_binom_posterior_plot.png')
    plt.savefig(posterior_path, bbox_inches='tight')
    image_paths['posterior_plot'] = posterior_path
    plt.close()
    
    return image_paths


def simple_binary_competition(data_sources, score_col = "score", entity_col = "entities", display = False, save_folder= "/tmp", **kwargs): 

    
    dataframes = [pd.read_csv(file) for file in data_sources]
    
    # Concatenate all DataFrames into one
    merged_data_source = pd.concat(dataframes, ignore_index=True).reset_index(drop = True)
    
    entities_idx, entities = pd.factorize(merged_data_source[entity_col], sort = True) 
    coords = {entity_col: entities}
    
    agg = (merged_data_source.groupby([entity_col]).agg(successes=(score_col, "sum"), n=(score_col, "size")).reset_index())
    model = build_model_simple_binary_competition(agg, coords, score_col)
    # model = build_model_simple_binary_competition(merged_data_source, coords, entities_idx, score_col)

    graph_path = render_graph_model(model, save_folder)

    with model:     
        trace = pm.sample(1000, tune = 1000)
    
    trace_path = os.path.join(os.path.dirname(graph_path), "trace.nc")
    az.to_netcdf(trace, trace_path)
    
    image_paths = simple_binary_competition_plot(trace, agg, graph_path, score_col, entity_col)
    image_paths["graph_model"] = graph_path

    # Optionally display plots
    if display:
        for img in image_paths.values():
            img_data = plt.imread(img)
            plt.figure()
            plt.imshow(img_data)
            plt.axis('off')
        plt.show()

    # Build resulting dict
    resulting_dict = {
        "trace": trace_path,
        "image_paths": image_paths,
        "model_builder_file": __file__, 
        "model_builder_function": inspect.currentframe().f_code.co_name,
    }

    return resulting_dict
