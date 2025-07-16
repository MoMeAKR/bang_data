import os 
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_entity_data_df(target_entity, num_entries=300, seed=92, max_days_ago=90):
    """
    Generate a pandas DataFrame simulating collected data for a specific entity (e.g., firm).
    The technical score is influenced by an entity-specific baseline and a random date.

    Args:
        target_entity (str): The name of the entity (e.g., firm) for which to generate data.
        num_entries (int): Number of data entries to generate.
        seed (int): Random seed for reproducibility.
        max_days_ago (int): Maximum number of days in the past for the date field.

    Returns:
        df (pd.DataFrame): Generated DataFrame with columns ['from', 'proposed_tech_score', 'date'].
        baseline (float): The baseline value used for the entity.
    """
    np.random.seed(seed)
    random.seed(seed)
    # Assign a Poisson-distributed baseline to the entity
    baseline = np.random.poisson(3.)
    data = []
    for _ in range(num_entries):
        proposed_tech_score = np.random.normal(0., 3.) + baseline
        # Ensure minimum score of 1.1
        proposed_tech_score = proposed_tech_score if proposed_tech_score >= 1. else 1.1
        days_ago = np.random.randint(0, max_days_ago)
        date = (datetime.today() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        data.append({
            "from": target_entity,
            "proposed_tech_score": proposed_tech_score,
            "date": date
        })
    df = pd.DataFrame(data)
    df['proposed_tech_score'] = pd.to_numeric(df['proposed_tech_score'], errors="coerce").astype(np.float64)
    return df, baseline

if __name__ == "__main__": 
    # Example usage: generate datasets for multiple entities
    entities = ['Capge', 'Inetum', 'Expleo']
    datasets = {}
    baselines = {}

    for i, entity in enumerate(entities):
        df, baseline = generate_entity_data_df(entity, num_entries=100, seed=29 + i)
        df.to_csv(os.path.join(os.path.dirname(__file__), entity.lower().strip() + ".csv"), index=False)
        # datasets[entity] = df
        # baselines[entity] = baseline
        print(f"Generated dataset for {entity} with baseline {baseline}")

