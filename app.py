import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dash.exceptions import PreventUpdate
import logging
import numpy as np
import json
import os
import threading
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'portfolio_app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the app with Bootstrap dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Investment Portfolio Projector"

# Global variables
portfolio = {}
dark_mode = True  # Default to dark mode

# File to store portfolio data
PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'portfolio.json')

def save_portfolio():
    """Save portfolio data to JSON file"""
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f)
        logger.info(f"Portfolio saved to {PORTFOLIO_FILE}")
    except Exception as e:
        logger.error(f"Error saving portfolio: {str(e)}")

def load_portfolio():
    """Load portfolio data from JSON file"""
    global portfolio
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r') as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    portfolio = loaded_data
                    logger.info(f"Portfolio loaded from {PORTFOLIO_FILE}")
                    # Update prices on load
                    update_stock_prices()
                else:
                    logger.error("Invalid portfolio data format")
                    portfolio = {}
        else:
            logger.info("No existing portfolio file found")
            portfolio = {}
    except Exception as e:
        logger.error(f"Error loading portfolio: {str(e)}")
        portfolio = {}

def update_stock_prices():
    """Update stock prices for all stocks in portfolio"""
    updated_count = 0
    for symbol in list(portfolio.keys()):
        try:
            current_price = get_stock_price(symbol)
            portfolio[symbol]['price'] = current_price
            updated_count += 1
        except Exception as e:
            logger.error(f"Error updating price for {symbol}: {str(e)}")
    
    if updated_count > 0:
        logger.info(f"Updated prices for {updated_count} stocks")
        save_portfolio()  # Save after updating prices

def auto_update_prices():
    """Background thread to update prices every 5 minutes during market hours"""
    while True:
        now = datetime.now()
        # Only update during market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
        if (now.weekday() < 5 and  # Monday-Friday
            ((now.hour == 9 and now.minute >= 30) or  # After 9:30 AM
             (now.hour > 9 and now.hour < 16) or  # 10 AM - 3:59 PM
             (now.hour == 16 and now.minute == 0))):  # Until 4:00 PM
            update_stock_prices()
        time.sleep(300)  # Wait 5 minutes

# Load portfolio on startup
load_portfolio()

# Start background price updates
update_thread = threading.Thread(target=auto_update_prices, daemon=True)
update_thread.start()

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Investment Portfolio Projector", className="text-center mb-4", style={'color': 'white'}),
            dbc.Switch(
                id="theme-switch",
                label="Dark Mode",
                value=dark_mode,
                className="mb-3"
            ),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Stock Symbol"),
                            dbc.Input(id="stock-input", type="text", placeholder="e.g., AAPL")
                        ]),
                        dbc.Col([
                            dbc.Label("Number of Shares"),
                            dbc.Input(id="shares-input", type="number", min=0, step="any")
                        ]),
                        dbc.Col([
                            dbc.Label("Monthly Contribution ($)"),
                            dbc.Input(id="stock-contribution", type="number", min=0, step="any", value=0)
                        ]),
                        dbc.Col([
                            dbc.Button("Add Stock", id="add-stock", color="primary", className="mt-4")
                        ])
                    ]),
                    html.Div(id="add-stock-output", className="mt-3"),
                    html.Div(id="portfolio-table", className="mt-4"),
                    html.Div(id="portfolio-return-display", className="mt-3")
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Projection Years"),
                            dbc.Input(id="years-input", type="number", min=1, max=50, step=1, value=25)
                        ], width=3),
                        dbc.Col([
                            dbc.Button("Calculate Projection", id="calculate-btn", color="success", className="mt-4")
                        ], width=3)
                    ]),
                    dcc.Graph(id="portfolio-graph", className="mt-4")
                ])
            ])
        ])
    ])
], fluid=True, className="p-5", id="main-container", style={'background-color': '#1a1a1a', 'min-height': '100vh'})

@app.callback(
    [Output("main-container", "style"),
     Output("portfolio-table", "children"),
     Output("portfolio-return-display", "children")],
    [Input("theme-switch", "value")],
    prevent_initial_call=False
)
def update_theme_and_initial_portfolio(dark_mode_enabled):
    # Update theme
    style = {
        'background-color': '#1a1a1a' if dark_mode_enabled else 'white',
        'color': 'white' if dark_mode_enabled else 'black',
        'min-height': '100vh'
    }
    
    # Return initial portfolio state
    return style, create_portfolio_table(), get_portfolio_return_display()

def create_portfolio_table():
    if not portfolio:
        return html.Div("No stocks in portfolio yet.", className="text-muted")
    
    # Calculate historical returns for each stock
    returns_data = {}
    portfolio_returns = []
    
    for symbol in portfolio:
        avg_return, min_return, max_return = calculate_historical_returns(symbol)
        if avg_return is not None:
            returns_data[symbol] = {
                'avg': avg_return,
                'min': min_return,
                'max': max_return
            }
            portfolio_returns.append(avg_return)
    
    # Calculate portfolio average return (weighted by position size)
    total_portfolio_value = sum(data['shares'] * data['price'] for data in portfolio.values())
    weighted_portfolio_return = 0
    total_monthly_contribution = sum(data.get('contribution', 0) for data in portfolio.values())
    
    if total_portfolio_value > 0:
        for symbol, data in portfolio.items():
            if symbol in returns_data:
                weight = (data['shares'] * data['price']) / total_portfolio_value
                weighted_portfolio_return += returns_data[symbol]['avg'] * weight
    
    # Create the table with historical returns
    table_header = [
        html.Thead([
            html.Tr([
                html.Th("Symbol"),
                html.Th("Shares"),
                html.Th("Current Price"),
                html.Th("Total Value"),
                html.Th("Monthly Contribution"),
                html.Th("Avg Annual Return (5Y)"),
                html.Th("Min/Max Annual Return"),
                html.Th("Action")
            ])
        ])
    ]
    
    rows = []
    for symbol, data in portfolio.items():
        returns = returns_data.get(symbol, {'avg': None, 'min': None, 'max': None})
        
        # Create shares input with current value
        shares_input = dbc.InputGroup(
            [
                dbc.Input(
                    id={'type': 'shares-input', 'symbol': symbol},
                    type="number",
                    min=0,
                    step="any",
                    value=data['shares'],
                    style={'width': '100px'}
                )
            ],
            size="sm"
        )
        
        # Create contribution input with current value
        contribution_input = dbc.InputGroup(
            [
                dbc.InputGroupText("$"),
                dbc.Input(
                    id={'type': 'contribution-input', 'symbol': symbol},
                    type="number",
                    min=0,
                    step=1,
                    value=data.get('contribution', 0),
                    style={'width': '100px'}
                )
            ],
            size="sm"
        )
        
        row = html.Tr([
            html.Td(symbol),
            html.Td(shares_input),
            html.Td(f"${data['price']:,.2f}"),
            html.Td(f"${data['shares'] * data['price']:,.2f}"),
            html.Td(contribution_input),
            html.Td(
                f"{returns['avg']:.1f}%" if returns['avg'] is not None else "N/A",
                style={'color': 'lime' if returns.get('avg', 0) > 0 else 'red'}
            ),
            html.Td(
                f"{returns['min']:.1f}% to {returns['max']:.1f}%" if returns['min'] is not None else "N/A"
            ),
            html.Td([
                dbc.Button(
                    "Remove",
                    id={'type': 'remove-stock', 'symbol': symbol},
                    color="danger",
                    size="sm",
                    className="me-2"
                )
            ])
        ])
        rows.append(row)
    
    # Add portfolio summary row
    summary_row = html.Tr([
        html.Td("Portfolio Total", style={'font-weight': 'bold'}),
        html.Td(""),
        html.Td(""),
        html.Td(f"${total_portfolio_value:,.2f}", style={'font-weight': 'bold'}),
        html.Td(f"${total_monthly_contribution:,.2f}", style={'font-weight': 'bold'}),
        html.Td(
            f"{weighted_portfolio_return:.1f}%",
            style={
                'font-weight': 'bold',
                'color': 'lime' if weighted_portfolio_return > 0 else 'red'
            }
        ),
        html.Td("Weighted Average", style={'font-style': 'italic'}),
        html.Td("")
    ])
    rows.append(summary_row)
    
    table_body = [html.Tbody(rows)]
    
    table_style = {
        'color': 'white' if dark_mode else 'black',
        'backgroundColor': '#343a40' if dark_mode else 'white'
    }
    
    return dbc.Table(
        table_header + table_body,
        bordered=True,
        hover=True,
        responsive=True,
        className="mt-3",
        style=table_style
    )

def get_portfolio_return_display():
    """Calculate and format the portfolio's weighted average return"""
    if not portfolio:
        return dbc.Alert(
            "Add stocks to see projected returns based on 5-year historical performance",
            color="info"
        )
    
    total_value = sum(data['shares'] * data['price'] for data in portfolio.values())
    weighted_return = 0
    total_monthly = sum(data.get('contribution', 0) for data in portfolio.values())
    
    for symbol, data in portfolio.items():
        avg_return, _, _ = calculate_historical_returns(symbol)
        if avg_return is not None:
            weight = (data['shares'] * data['price']) / total_value
            weighted_return += avg_return * weight
    
    return dbc.Alert(
        [
            html.Strong("Portfolio Expected Return: "),
            f"{weighted_return:.1f}% ",
            html.Small("(based on 5-year historical weighted average)", className="text-muted"),
            html.Br(),
            html.Strong("Total Monthly Contribution: "),
            f"${total_monthly:.2f}"
        ],
        color="success" if weighted_return > 0 else "danger"
    )

@app.callback(
    [Output("portfolio-table", "children", allow_duplicate=True),
     Output("portfolio-return-display", "children", allow_duplicate=True),
     Output("portfolio-graph", "figure", allow_duplicate=True)],
    [Input("add-stock", "n_clicks"),
     Input({'type': 'contribution-input', 'symbol': ALL}, 'value'),
     Input({'type': 'shares-input', 'symbol': ALL}, 'value'),
     Input({'type': 'remove-stock', 'symbol': ALL}, "n_clicks")],
    [State({'type': 'contribution-input', 'symbol': ALL}, 'id'),
     State({'type': 'shares-input', 'symbol': ALL}, 'id'),
     State("stock-input", "value"),
     State("shares-input", "value"),
     State("stock-contribution", "value"),
     State("years-input", "value"),
     State("theme-switch", "value")],
    prevent_initial_call=True
)
def update_portfolio(add_clicks, contribution_values, shares_values, remove_clicks,
                    contribution_ids, shares_ids, symbol, new_shares, initial_contribution,
                    years, dark_mode_enabled):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger = ctx.triggered[0]['prop_id']
    
    try:
        # Handle contribution updates
        if 'contribution-input' in trigger:
            for contrib_id, value in zip(contribution_ids, contribution_values):
                symbol = contrib_id['symbol']
                if symbol in portfolio:
                    portfolio[symbol]['contribution'] = float(value) if value is not None else 0
            save_portfolio()
        
        # Handle shares updates
        elif 'shares-input' in trigger:
            for shares_id, value in zip(shares_ids, shares_values):
                symbol = shares_id['symbol']
                if symbol in portfolio and value is not None and value > 0:
                    portfolio[symbol]['shares'] = float(value)
            save_portfolio()
        
        # Handle add stock
        elif "add-stock" in trigger:
            if symbol and new_shares:
                try:
                    current_price = get_stock_price(symbol.upper().strip())
                    portfolio[symbol.upper().strip()] = {
                        'shares': float(new_shares),
                        'price': current_price,
                        'contribution': float(initial_contribution or 0)
                    }
                    save_portfolio()
                except ValueError as e:
                    logger.error(str(e))
        
        # Handle remove stock
        elif "remove-stock" in trigger:
            remove_dict = json.loads(trigger.split('.')[0])
            symbol_to_remove = remove_dict["symbol"]
            if symbol_to_remove in portfolio:
                del portfolio[symbol_to_remove]
                save_portfolio()
        
        # Update graph if years is available
        if years:
            fig = update_projection(None, years, dark_mode_enabled)
        else:
            fig = go.Figure()
        
        return create_portfolio_table(), get_portfolio_return_display(), fig
        
    except Exception as e:
        logger.error(f"Error updating portfolio: {str(e)}")
        raise PreventUpdate

@app.callback(
    Output("portfolio-graph", "figure"),
    [Input("calculate-btn", "n_clicks"),
     Input("years-input", "value"),
     Input("theme-switch", "value")],
    prevent_initial_call=True
)
def update_projection(n_clicks, years, dark_mode_enabled):
    if not years or not portfolio:
        return go.Figure()  # Return empty figure if no data
    
    try:
        # Calculate historical values first
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Get historical data for each stock
        historical_data = {}
        current_total = 0
        monthly_contributions = 0
        
        for symbol, data in portfolio.items():
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(start=start_date, end=end_date)
                if not hist.empty:
                    historical_data[symbol] = hist['Close'] * data['shares']
                    current_total += data['shares'] * data['price']
                    monthly_contributions += data.get('contribution', 0)
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
        
        if not historical_data:
            return go.Figure()
        
        # Combine historical data
        historical_dates = pd.date_range(start_date, end_date, freq='D')
        historical_values = pd.Series(0, index=historical_dates)
        
        for symbol_data in historical_data.values():
            historical_values = historical_values.add(
                symbol_data.reindex(historical_dates).fillna(method='ffill'),
                fill_value=0
            )
        
        # Calculate weighted average return for the portfolio
        total_value = sum(data['shares'] * data['price'] for data in portfolio.values())
        weighted_return = 0
        
        for symbol, data in portfolio.items():
            avg_return, _, _ = calculate_historical_returns(symbol)
            if avg_return is not None:
                weight = (data['shares'] * data['price']) / total_value
                weighted_return += avg_return * weight
        
        if weighted_return == 0:
            weighted_return = 8.0  # Default if no historical data available
        
        # Project future values using standard formula
        future_dates = pd.date_range(end_date, end_date + timedelta(days=365*years), freq='M')
        future_values = calculate_future_value(
            principal=current_total,
            monthly_payment=monthly_contributions,
            annual_rate=weighted_return,
            years=years
        )
        
        # Create the graph
        fig = go.Figure()
        
        # Add historical values
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_values,
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Add projected values
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_values[1:],  # Skip first value as it's the current value
            name='Projected',
            line=dict(color='green', dash='dash')
        ))
        
        # Calculate key metrics
        total_contributions = monthly_contributions * len(future_dates)
        final_value = future_values[-1]
        initial_value = current_total
        investment_return = ((final_value - initial_value - total_contributions) / 
                           (initial_value + total_contributions) * 100)
        
        # Update layout with key information
        title_text = (
            f'Portfolio Value Over Time<br>'
            f'<span style="font-size: 12px;">'
            f'Initial Value: ${initial_value:,.2f} | '
            f'Final Value: ${final_value:,.2f}<br>'
            f'Monthly Contribution: ${monthly_contributions:,.2f} | '
            f'Total Contributions: ${total_contributions:,.2f}<br>'
            f'Investment Return: {investment_return:.1f}% | '
            f'Annual Return Rate: {weighted_return:.1f}%'
            f'</span>'
        )
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_dark' if dark_mode_enabled else 'plotly',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, t=100, b=0)
        )
        
        # Format y-axis as currency
        fig.update_layout(yaxis_tickformat='$,.0f')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating graph: {str(e)}")
        return go.Figure()  # Return empty figure on error

def get_stock_price(symbol):
    """Safely retrieve the current stock price"""
    try:
        # Try different variations of the symbol for mutual funds
        symbols_to_try = [
            symbol,
            symbol + '.MF',  # Mutual fund suffix
            symbol + '.O',   # NASDAQ suffix
            symbol + '.X'    # Some mutual fund suffix
        ]
        
        for sym in symbols_to_try:
            try:
                stock = yf.Ticker(sym)
                hist = stock.history(period="1d")
                
                if not hist.empty and 'Close' in hist.columns:
                    latest_price = hist['Close'].iloc[-1] if len(hist) > 0 else None
                    
                    if latest_price is not None and not pd.isna(latest_price):
                        logger.info(f"Successfully found price for {sym}")
                        return latest_price
            except:
                continue
        
        # If we get here, none of the symbol variations worked
        # Try to get the info to provide a better error message
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if 'quoteType' in info:
            quote_type = info['quoteType']
            if quote_type == 'MUTUALFUND':
                raise ValueError(f"'{symbol}' is a mutual fund that is not supported by Yahoo Finance. "
                               "Try checking the fund company's website directly.")
            else:
                raise ValueError(f"No price data available for {symbol} (Type: {quote_type})")
        else:
            raise ValueError(f"Could not find {symbol}. Please verify the symbol is correct.")
            
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {str(e)}")
        if "mutual fund" in str(e).lower():
            raise ValueError(f"'{symbol}' appears to be a mutual fund. Currently, we only support stocks and ETFs. "
                           "For mutual funds, please check your fund company's website directly.")
        else:
            raise ValueError(f"Could not get price for {symbol}. Please verify the symbol is correct. "
                           "Note: We currently support stocks and ETFs only.")

def calculate_historical_returns(symbol, lookback_years=5):
    """Calculate historical returns for a given stock"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * lookback_years)
        
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.warning(f"No historical data found for {symbol}")
            return None, None, None
            
        if 'Close' not in hist.columns:
            logger.warning(f"No price data found for {symbol}")
            return None, None, None
            
        # Calculate annual returns
        annual_returns = []
        
        # Split data into years and calculate return for each year
        hist['Year'] = hist.index.year
        years = hist['Year'].unique()
        
        for year in years:
            year_data = hist[hist['Year'] == year]
            if len(year_data) >= 2:  # Need at least 2 points for return calculation
                year_return = ((year_data['Close'].iloc[-1] / year_data['Close'].iloc[0]) - 1) * 100
                annual_returns.append(year_return)
        
        if not annual_returns:
            logger.warning(f"Insufficient data to calculate returns for {symbol}")
            return None, None, None
            
        # Calculate statistics
        avg_return = np.mean(annual_returns)
        max_return = np.max(annual_returns)
        min_return = np.min(annual_returns)
        
        return avg_return, min_return, max_return
        
    except Exception as e:
        logger.error(f"Error calculating returns for {symbol}: {str(e)}")
        return None, None, None

def calculate_future_value(principal, monthly_payment, annual_rate, years):
    """
    Calculate future value using compound interest with regular contributions
    
    Args:
        principal: Initial investment
        monthly_payment: Monthly contribution amount
        annual_rate: Annual interest rate (as percentage)
        years: Number of years
        
    Returns:
        List of monthly values over the investment period
    """
    # Convert annual rate to monthly
    monthly_rate = (1 + annual_rate/100) ** (1/12) - 1
    num_months = years * 12
    
    # Calculate values for each month
    values = []
    for t in range(num_months + 1):
        # Standard compound interest formula with regular contributions:
        # FV = P(1 + r)^t + PMT × [((1 + r)^t - 1) / r]
        if monthly_rate > 0:
            fv = principal * (1 + monthly_rate)**t + \
                 monthly_payment * (((1 + monthly_rate)**t - 1) / monthly_rate)
        else:
            # If rate is 0, just sum principal and contributions
            fv = principal + (monthly_payment * t)
        values.append(fv)
    
    return values

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)
