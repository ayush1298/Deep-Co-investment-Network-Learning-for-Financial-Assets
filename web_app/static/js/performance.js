// static/js/performance.js
class PerformanceAnalysis {
    constructor() {
        this.loadPerformanceData();
    }

    async loadPerformanceData() {
        try {
            const response = await fetch('/api/performance');
            const data = await response.json();
            this.renderMarketCapChart(data.market_cap_comparison);
            this.renderDensityChart(data.investment_density);
        } catch (error) {
            console.error('Error loading performance data:', error);
        }
    }

    renderMarketCapChart(data) {
        const ctx = document.getElementById('marketCapChart').getContext('2d');
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['DeepCNL', 'PCC Baseline'],
                datasets: [{
                    label: 'Average Market Cap ($B)',
                    data: [data.deepcnl, data.pcc],
                    backgroundColor: [
                        'rgba(52, 152, 219, 0.8)',
                        'rgba(231, 76, 60, 0.8)'
                    ],
                    borderColor: [
                        'rgba(52, 152, 219, 1)',
                        'rgba(231, 76, 60, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Average Market Cap of Top-Ranked Firms',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Market Cap (Billions USD)'
                        }
                    }
                }
            }
        });
    }

    renderDensityChart(data) {
        const ctx = document.getElementById('densityChart').getContext('2d');
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['XLG (Top 50)', 'OEX (Top 100)', 'IWL (Top 200)'],
                datasets: [
                    {
                        label: 'DeepCNL',
                        data: [data.xlg.deepcnl, data.oex.deepcnl, data.iwl.deepcnl],
                        borderColor: 'rgba(52, 152, 219, 1)',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 3,
                        fill: true
                    },
                    {
                        label: 'PCC',
                        data: [data.xlg.pcc, data.oex.pcc, data.iwl.pcc],
                        borderColor: 'rgba(231, 76, 60, 1)',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        borderWidth: 3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Investment Density in Top ETFs',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Edge Density'
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 6,
                        hoverRadius: 8
                    }
                }
            }
        });
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new PerformanceAnalysis();
});