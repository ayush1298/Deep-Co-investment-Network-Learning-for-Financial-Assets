# app.py
from flask import Flask, render_template, jsonify, request
import json
import os

app = Flask(__name__)

# Load precomputed data
def load_data():
    """Load all precomputed data files"""
    data_dir = 'data'
    
    with open(os.path.join(data_dir, 'network_data.json'), 'r') as f:
        network_data = json.load(f)
    
    with open(os.path.join(data_dir, 'stock_rankings.json'), 'r') as f:
        rankings_data = json.load(f)
    
    with open(os.path.join(data_dir, 'performance_metrics.json'), 'r') as f:
        performance_data = json.load(f)
    
    return network_data, rankings_data, performance_data

NETWORK_DATA, RANKINGS_DATA, PERFORMANCE_DATA = load_data()

@app.route('/')
def index():
    """Landing page with key performance indicators"""
    kpis = {
        'hit_ratio': 57.14,
        'market_cap': 223.1,
        'benchmark_improvement': 3.3
    }
    return render_template('index.html', kpis=kpis)

@app.route('/network')
def network():
    """Network explorer page"""
    years = list(range(2010, 2017))
    return render_template('network.html', years=years)

@app.route('/rankings')
def rankings():
    """Stock rankings and prediction page"""
    years = list(range(2010, 2017))
    return render_template('rankings.html', years=years)

@app.route('/performance')
def performance():
    """Performance and financial analysis page"""
    return render_template('performance.html')

# API Endpoints
@app.route('/api/network/<int:year>')
def get_network_data(year):
    """Get network data for a specific year"""
    if str(year) in NETWORK_DATA:
        return jsonify(NETWORK_DATA[str(year)])
    return jsonify({'error': 'Year not found'}), 404

@app.route('/api/rankings/<int:year>')
def get_rankings_data(year):
    """Get stock rankings for a specific year"""
    if str(year) in RANKINGS_DATA:
        return jsonify(RANKINGS_DATA[str(year)])
    return jsonify({'error': 'Year not found'}), 404

@app.route('/api/performance')
def get_performance_data():
    """Get performance comparison data"""
    return jsonify(PERFORMANCE_DATA)

@app.route('/api/search_stock')
def search_stock():
    """Search for a stock in the network"""
    ticker = request.args.get('ticker', '').upper()
    year = request.args.get('year', '2010')
    
    if str(year) in NETWORK_DATA and ticker:
        network = NETWORK_DATA[str(year)]
        for node in network['nodes']:
            if node['id'] == ticker:
                return jsonify(node)
    
    return jsonify({'error': 'Stock not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)