// static/js/rankings.js
class StockRankings {
    constructor() {
        this.currentYear = 2012;
        this.bindEvents();
        this.loadRankings(this.currentYear);
    }

    bindEvents() {
        document.getElementById('yearSelectRanking').addEventListener('change', (e) => {
            this.currentYear = parseInt(e.target.value);
            this.loadRankings(this.currentYear);
            this.updateYearDisplay();
        });
    }

    async loadRankings(year) {
        try {
            const response = await fetch(`/api/rankings/${year}`);
            const data = await response.json();
            this.renderRankings(data);
            this.updateHitRatio(data);
        } catch (error) {
            console.error('Error loading rankings:', error);
        }
    }

    renderRankings(data) {
        const tbody = document.getElementById('rankingsBody');
        tbody.innerHTML = '';

        data.top_stocks.forEach((stock, index) => {
            const row = document.createElement('tr');
            const isPredicted = data.deepcnl_predictions.includes(stock.ticker);
            
            if (isPredicted) {
                row.classList.add('predicted-stock');
            }

            row.innerHTML = `
                <td>${index + 1}</td>
                <td><strong>${stock.ticker}</strong></td>
                <td>${stock.company_name}</td>
                <td>${stock.return.toFixed(2)}%</td>
                <td>
                    ${isPredicted ? 
                        '<i class="fas fa-check-circle text-success"></i> Predicted' : 
                        '<i class="fas fa-times-circle text-muted"></i> Not Predicted'
                    }
                </td>
            `;

            tbody.appendChild(row);
        });
    }

    updateHitRatio(data) {
        const hitRatio = data.hit_ratio;
        const hitRatioValue = document.getElementById('hitRatioValue');
        hitRatioValue.textContent = `${data.deepcnl_hits} / 10`;
        
        // Add animation
        hitRatioValue.style.transform = 'scale(1.1)';
        setTimeout(() => {
            hitRatioValue.style.transform = 'scale(1)';
        }, 200);
    }

    updateYearDisplay() {
        document.getElementById('selectedYear').textContent = this.currentYear;
        document.getElementById('rankingYear').textContent = this.currentYear;
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new StockRankings();
});