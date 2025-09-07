# generate_web_data.py
import json
import numpy as np
from stock_network_analysis import *

def generate_sample_data():
    """Generate sample data for the web application"""
    
    # Initialize your existing components
    datatool = Data_util(TICKER_NUM, WINDOW, FEATURE_NUM, DATA_PATH, SPY_PATH)
    experiment = Experimental_platform(datatool)
    
    network_data = {}
    rankings_data = {}
    
    for seed in range(7):  # 2010-2016
        year = 2010 + seed
        
        # Generate network data
        train_period, test_period = experiment.period_generator(seed)
        train_x = datatool.load_x(train_period)
        train_y = datatool.load_y(train_period)
        
        # Learn network
        network = experiment.deep_CNL('igo', train_x, train_y, RARE_RATIO)
        
        # Convert to web format
        nodes = []
        links = []
        
        for node in network.nodes():
            degree = network.degree(node)
            nodes.append({
                'id': node,
                'name': f"{node} Inc.",  # You can add real company names
                'degree': degree,
                'market_cap': np.random.randint(10000, 500000),  # Sample data
                'top_connections': [
                    {'ticker': neighbor, 'weight': np.random.random()}
                    for neighbor in list(network.neighbors(node))[:3]
                ]
            })
        
        for edge in network.edges(data=True):
            links.append({
                'source': edge[0],
                'target': edge[1],
                'weight': edge[2].get('weight', 1.0)
            })
        
        network_data[str(year)] = {
            'nodes': nodes,
            'links': links
        }
        
        # Generate rankings data (you can extract from your real results)
        rankings_data[str(year)] = {
            'top_stocks': [
                {'ticker': 'NFLX', 'company_name': 'Netflix Inc.', 'return': 298.73},
                # Add more stocks...
            ],
            'deepcnl_predictions': ['NFLX', 'MU', 'BBY'],
            'deepcnl_hits': 6,
            'hit_ratio': 60.0
        }
    
    # Performance data
    performance_data = {
        'market_cap_comparison': {
            'deepcnl': 223.1,
            'pcc': 67.0
        },
        'investment_density': {
            'xlg': {'deepcnl': 0.45, 'pcc': 0.32},
            'oex': {'deepcnl': 0.38, 'pcc': 0.41},
            'iwl': {'deepcnl': 0.28, 'pcc': 0.35}
        }
    }
    
    # Save data files
    with open('data/network_data.json', 'w') as f:
        json.dump(network_data, f, indent=2)
    
    with open('data/stock_rankings.json', 'w') as f:
        json.dump(rankings_data, f, indent=2)
    
    with open('data/performance_metrics.json', 'w') as f:
        json.dump(performance_data, f, indent=2)

if __name__ == "__main__":
    generate_sample_data()