# Portfolio Tracker

A modern web application to track and project your investment portfolio value using real-time stock data.

## Features

- Real-time stock data integration using Yahoo Finance API
- Interactive Plotly graphs showing historical and projected portfolio value
- Modern and responsive UI using Dash and Bootstrap
- Portfolio value projection based on custom time duration
- Easy stock addition with automatic price updates

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://127.0.0.1:8050`

## Usage

1. Enter a stock symbol (e.g., AAPL, GOOGL) and the number of shares you own
2. Click "Add to Portfolio" to add the stock to your portfolio
3. Enter the number of years you want to project your portfolio value
4. Click "Calculate Projection" to see the projected growth

The graph will show both historical data (solid line) and projected growth (dashed line) based on an 8% annual return assumption.
