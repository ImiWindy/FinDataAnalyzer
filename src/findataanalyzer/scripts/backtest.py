"""
Main backtesting script to run the advanced ML-based strategy on real data,
calculate performance metrics, and save the results to a JSON file.
"""
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import quantstats as qs

# --- Standard Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import Core Components ---
from findataanalyzer.data.downloader import download_data
from findataanalyzer.strategy import MLStrategy
from findataanalyzer.trading import RiskManager

# --- Backtest Engine (included here for script simplicity) ---
class Portfolio:
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.positions_value = 0.0
        self.total_value = initial_cash
        self.history = []

    def update_value(self, current_prices: Dict[str, float], timestamp: datetime):
        self.positions_value = sum(self.positions.get(symbol, 0) * current_prices.get(symbol, 0) for symbol in self.positions)
        self.total_value = self.cash + self.positions_value
        self.history.append({'timestamp': timestamp, 'total_value': self.total_value})

    def execute_trade(self, symbol: str, quantity: float, price: float, side: str, timestamp: datetime):
        trade_cost = quantity * price
        if side == 'buy':
            if self.cash < trade_cost: return False
            self.cash -= trade_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        elif side == 'sell':
            if self.positions.get(symbol, 0, 0) < quantity: return False
            self.cash += trade_cost
            self.positions[symbol] -= quantity
            if self.positions[symbol] == 0: del self.positions[symbol]
        return True

class BacktestEngine:
    def __init__(self, portfolio: Portfolio, strategy, risk_manager, config):
        self.portfolio = portfolio
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config
        self.trade_log = []
        self.open_trades = {}

    def run(self, market_data: Dict[str, pd.DataFrame], symbol: str):
        fastest_tf_key = self.strategy.trigger_timeframe
        data = market_data[fastest_tf_key]
        
        for i in range(60, len(data)):
            timestamp = data.index[i]
            current_price = data['close'].iloc[i]
            
            # 1. Update portfolio value
            self.portfolio.update_value({symbol: current_price}, timestamp)

            # 2. Check for closing open trades
            trade_to_close = None
            if symbol in self.open_trades:
                trade = self.open_trades[symbol]
                if current_price <= trade['stop_loss']:
                    reason = 'Stop Loss'
                    trade_to_close = (trade, reason)
                elif current_price >= trade['take_profit']:
                    reason = 'Take Profit'
                    trade_to_close = (trade, reason)
            
            if trade_to_close:
                trade, reason = trade_to_close
                self.portfolio.execute_trade(symbol, trade['quantity'], current_price, 'sell', timestamp)
                self.trade_log.append({**trade, 'exit_price': current_price, 'exit_time': timestamp, 'exit_reason': reason})
                del self.open_trades[symbol]

            # 3. Check for new signals (only if no position is open)
            if symbol not in self.open_trades:
                historical_window = {tf: df.loc[df.index <= timestamp] for tf, df in market_data.items()}
                signals = self.strategy.generate_signals(historical_window)

                for signal in signals:
                    if signal.action == 'BUY':
                        quantity = self.risk_manager.calculate_position_size(
                            self.portfolio.cash, signal.entry_price, signal.stop_loss_price,
                            risk_percentage=self.config['risk_pct']
                        )
                        if quantity > 0 and self.portfolio.execute_trade(symbol, quantity, signal.entry_price, 'buy', timestamp):
                            trade_info = {
                                'symbol': symbol, 'entry_price': signal.entry_price, 'entry_time': timestamp,
                                'stop_loss': signal.stop_loss_price, 'take_profit': signal.take_profit_price,
                                'quantity': quantity, 'rationale': signal.rationale
                            }
                            self.open_trades[symbol] = trade_info
                            self.trade_log.append(trade_info) # Log entry

# --- Main Execution Logic ---
def check_and_prepare_env(symbols, timeframes, days):
    logger.info("Checking environment...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data_dir = Path("data")
    if not all((data_dir / f"{s}_{tf}.csv").exists() for s in symbols for tf in timeframes):
        logger.info("Data missing, starting download...")
        download_data(symbols, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), timeframes, data_dir)
    
    model_path = Path("models/xgboost_model.joblib")
    if not model_path.exists():
        logger.info("XGBoost model not found, creating one...")
        import os
        os.system("python create_test_model.py")

def load_data(symbol, timeframes):
    data_dir = Path("data")
    return {tf: pd.read_csv(data_dir / f"{symbol}_{tf}.csv", index_col=0, parse_dates=True) for tf in timeframes}

def main():
    config = {
        "symbols": ["GC=F", "DIA"],
        "timeframes": {"1h": "60m", "15m": "15m", "5m": "5m"},
        "setup_tfs": ["1h", "15m"], "trigger_tf": "5m",
        "backtest_days": 365, "risk_pct": 1.5,
        "model_path": "models/xgboost_model.joblib",
        "prediction_threshold": 0.55, "risk_reward_ratio": 1.5,
    }
    check_and_prepare_env(config['symbols'], config['timeframes'], config['backtest_days'])
    
    full_results = {"backtest_summary": {}}
    
    for symbol in config['symbols']:
        logger.info(f"--- Running Backtest for {symbol} ---")
        market_data = load_data(symbol, config['timeframes'].keys())
        
        portfolio = Portfolio()
        risk_manager = RiskManager(config={})
        strategy = MLStrategy({
            "symbol": symbol, "model_path": config['model_path'],
            "prediction_threshold": config['prediction_threshold'], 
            "risk_reward_ratio": config['risk_reward_ratio'],
            "setup_timeframes": config['setup_tfs'],
            "trigger_timeframe": config['trigger_tf']
        })
        
        engine = BacktestEngine(portfolio, strategy, risk_manager, config)
        engine.run(market_data, symbol)
        
        # --- Performance Reporting ---
        returns = pd.Series([h['total_value'] for h in portfolio.history],
                              index=[h['timestamp'] for h in portfolio.history]).pct_change().dropna()
        
        if not returns.empty:
            metrics = {
                'sharpe': qs.stats.sharpe(returns),
                'max_drawdown': qs.stats.max_drawdown(returns),
                'total_return': qs.stats.comp(returns) * 100
            }
        else:
            metrics = {'sharpe': 0, 'max_drawdown': 0, 'total_return': 0}

        full_results['backtest_summary'][symbol] = {
            "metrics": metrics,
            "total_trades": len(engine.trade_log),
            "final_portfolio_value": portfolio.total_value,
            "trade_log": [{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in t.items()} for t in engine.trade_log]
        }
    
    # --- Save final results ---
    output_path = Path("backtest_results.json")
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    logger.info(f"Backtest finished. Results saved to {output_path}")
    print(json.dumps(full_results['backtest_summary'], indent=2))

if __name__ == "__main__":
    main() 