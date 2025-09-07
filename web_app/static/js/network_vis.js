// static/js/network_viz.js
class NetworkVisualization {
    constructor() {
        this.svg = null;
        this.simulation = null;
        this.currentData = null;
        this.width = 0;
        this.height = 0;
        this.init();
        this.bindEvents();
    }

    init() {
        const container = d3.select('#networkViz');
        const containerNode = container.node();
        this.width = containerNode.clientWidth;
        this.height = containerNode.clientHeight;

        this.svg = container.append('svg')
            .attr('width', this.width)
            .attr('height', this.height);

        this.svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 13)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 13)
            .attr('markerHeight', 13)
            .attr('xoverflow', 'visible')
            .append('svg:path')
            .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
            .attr('fill', '#999')
            .style('stroke', 'none');

        this.loadNetworkData(2012);
    }

    bindEvents() {
        document.getElementById('yearSelect').addEventListener('change', (e) => {
            this.loadNetworkData(parseInt(e.target.value));
        });

        document.getElementById('searchBtn').addEventListener('click', () => {
            this.searchStock();
        });

        document.getElementById('stockSearch').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.searchStock();
            }
        });
    }

    async loadNetworkData(year) {
        document.getElementById('loading').style.display = 'block';
        
        try {
            const response = await fetch(`/api/network/${year}`);
            const data = await response.json();
            this.currentData = data;
            this.renderNetwork(data);
        } catch (error) {
            console.error('Error loading network data:', error);
        } finally {
            document.getElementById('loading').style.display = 'none';
        }
    }

    renderNetwork(data) {
        this.svg.selectAll('*').remove();

        // Create scales
        const linkScale = d3.scaleLinear()
            .domain(d3.extent(data.links, d => d.weight))
            .range([1, 8]);

        const nodeScale = d3.scaleLinear()
            .domain(d3.extent(data.nodes, d => d.degree))
            .range([5, 20]);

        // Create force simulation
        this.simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2));

        // Create links
        const link = this.svg.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(data.links)
            .enter().append('line')
            .attr('stroke-width', d => linkScale(d.weight))
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6);

        // Create nodes
        const node = this.svg.append('g')
            .attr('class', 'nodes')
            .selectAll('circle')
            .data(data.nodes)
            .enter().append('circle')
            .attr('r', d => nodeScale(d.degree))
            .attr('fill', d => this.getNodeColor(d))
            .call(d3.drag()
                .on('start', (event, d) => this.dragstarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragended(event, d)));

        // Add labels
        const labels = this.svg.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(data.nodes.filter(d => d.degree > 10)) // Only show labels for high-degree nodes
            .enter().append('text')
            .text(d => d.id)
            .attr('font-size', '10px')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em');

        // Add click events
        node.on('click', (event, d) => {
            this.showStockInfo(d);
            this.highlightNode(d);
        });

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    }

    getNodeColor(node) {
        if (node.degree > 20) return '#e74c3c'; // High degree - red
        if (node.degree > 10) return '#f39c12'; // Medium degree - orange
        return '#3498db'; // Low degree - blue
    }

    showStockInfo(node) {
        const infoDiv = document.getElementById('stockInfo');
        const nameDiv = document.getElementById('stockName');
        const detailsDiv = document.getElementById('stockDetails');

        nameDiv.textContent = `${node.name} (${node.id})`;
        
        let detailsHTML = `
            <p><strong>Degree:</strong> ${node.degree}</p>
            <p><strong>Market Cap:</strong> $${(node.market_cap / 1000).toFixed(1)}B</p>
            <h6>Top Connections:</h6>
            <ul>
        `;

        node.top_connections.forEach(conn => {
            detailsHTML += `<li>${conn.ticker}: ${conn.weight.toFixed(3)}</li>`;
        });

        detailsHTML += '</ul>';
        detailsDiv.innerHTML = detailsHTML;
        infoDiv.style.display = 'block';
    }

    highlightNode(selectedNode) {
        this.svg.selectAll('circle')
            .attr('stroke', d => d.id === selectedNode.id ? '#2c3e50' : 'none')
            .attr('stroke-width', d => d.id === selectedNode.id ? 3 : 0);
    }

    async searchStock() {
        const ticker = document.getElementById('stockSearch').value.toUpperCase();
        const year = document.getElementById('yearSelect').value;

        if (!ticker) return;

        try {
            const response = await fetch(`/api/search_stock?ticker=${ticker}&year=${year}`);
            const data = await response.json();
            
            if (data.error) {
                alert('Stock not found in the network');
                return;
            }

            this.showStockInfo(data);
            this.highlightNode(data);
            
            // Center the view on the selected node
            const node = this.svg.selectAll('circle').data().find(d => d.id === ticker);
            if (node) {
                const transform = d3.zoomIdentity
                    .translate(this.width / 2 - node.x, this.height / 2 - node.y)
                    .scale(1.5);
                
                this.svg.transition().duration(750).call(
                    d3.zoom().transform, transform
                );
            }
        } catch (error) {
            console.error('Error searching stock:', error);
        }
    }

    dragstarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new NetworkVisualization();
});