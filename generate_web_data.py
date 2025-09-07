import json
import numpy as np
import os
from stock_network_analysis import *

def generate_sample_data():
    """Generate sample data for the web application"""
    
    # Create data directory if it doesn't exist
    os.makedirs('web_app/data', exist_ok=True)
    
    # Since we can't run the full model without actual data, let's create sample data
    network_data = {}
    rankings_data = {}
    
    # Sample tickers for demonstration
    sample_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'BAC', 'JPM']
    
    for seed in range(7):  # 2010-2016
        year = 2010 + seed
        
        # Generate sample network data
        nodes = []
        links = []
        
        for i, ticker in enumerate(sample_tickers):
            degree = np.random.randint(5, 25)
            nodes.append({
                'id': ticker,
                'name': f"{ticker} Inc.",
                'degree': degree,
                'market_cap': np.random.randint(50000, 500000),
                'top_connections': [
                    {'ticker': sample_tickers[(i+j+1) % len(sample_tickers)], 'weight': np.random.random()}
                    for j in range(3)
                ]
            })
        
        # Generate sample links
        for i in range(len(sample_tickers)):
            for j in range(i+1, min(i+4, len(sample_tickers))):
                links.append({
                    'source': sample_tickers[i],
                    'target': sample_tickers[j],
                    'weight': np.random.random()
                })
        
        network_data[str(year)] = {
            'nodes': nodes,
            'links': links
        }
        
        # Generate sample rankings data
        top_stocks = [
            {'ticker': 'NFLX', 'company_name': 'Netflix Inc.', 'return': 298.73},
            {'ticker': 'NVDA', 'company_name': 'NVIDIA Corporation', 'return': 245.67},
            {'ticker': 'TSLA', 'company_name': 'Tesla Inc.', 'return': 198.45},
            {'ticker': 'AAPL', 'company_name': 'Apple Inc.', 'return': 156.78},
            {'ticker': 'AMZN', 'company_name': 'Amazon.com Inc.', 'return': 134.56},
            {'ticker': 'MSFT', 'company_name': 'Microsoft Corporation', 'return': 123.45},
            {'ticker': 'GOOGL', 'company_name': 'Alphabet Inc.', 'return': 112.34},
            {'ticker': 'META', 'company_name': 'Meta Platforms Inc.', 'return': 98.76},
            {'ticker': 'BAC', 'company_name': 'Bank of America Corp', 'return': 87.65},
            {'ticker': 'JPM', 'company_name': 'JPMorgan Chase & Co.', 'return': 76.54}
        ]
        
        # Randomize predictions for different years
        predicted_count = np.random.randint(4, 8)
        deepcnl_predictions = np.random.choice([stock['ticker'] for stock in top_stocks], 
                                               size=predicted_count, replace=False).tolist()
        
        rankings_data[str(year)] = {
            'top_stocks': top_stocks,
            'deepcnl_predictions': deepcnl_predictions,
            'deepcnl_hits': predicted_count,
            'hit_ratio': (predicted_count / 10) * 100
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
    with open('web_app/data/network_data.json', 'w') as f:
        json.dump(network_data, f, indent=2)
    
    with open('web_app/data/stock_rankings.json', 'w') as f:
        json.dump(rankings_data, f, indent=2)
    
    with open('web_app/data/performance_metrics.json', 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    print("Sample data generated successfully!")
    print("Files created:")
    print("- web_app/data/network_data.json")
    print("- web_app/data/stock_rankings.json") 
    print("- web_app/data/performance_metrics.json")

if __name__ == "__main__":
    generate_sample_data()