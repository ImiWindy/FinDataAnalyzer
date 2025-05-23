"""ماژول مدیریت معاملات.

این ماژول مسئول مدیریت معاملات و استراتژی‌های معاملاتی است.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd


class TradeManager:
    """مدیریت معاملات."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه مدیر معاملات.
        
        Args:
            config: تنظیمات مدیریت معاملات
        """
        self.config = config
        self.trades_dir = Path(config['trades_dir'])
        self.strategies_dir = Path(config['strategies_dir'])
        
        # ایجاد دایرکتوری‌های مورد نیاز
        self._create_directories()
        
        # بارگذاری استراتژی‌ها
        self.strategies = self._load_strategies()
    
    def _create_directories(self) -> None:
        """ایجاد دایرکتوری‌های مورد نیاز."""
        for directory in [self.trades_dir, self.strategies_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری استراتژی‌های معاملاتی.
        
        Returns:
            دیکشنری استراتژی‌ها
        """
        strategies = {}
        for strategy_file in self.strategies_dir.glob("*.json"):
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
                strategies[strategy['name']] = strategy
        return strategies
    
    def add_strategy(self, name: str, parameters: Dict[str, Any]) -> None:
        """
        افزودن استراتژی معاملاتی.
        
        Args:
            name: نام استراتژی
            parameters: پارامترهای استراتژی
        """
        strategy = {
            'name': name,
            'parameters': parameters,
            'created_at': datetime.now().isoformat()
        }
        
        strategy_file = self.strategies_dir / f"{name}.json"
        with open(strategy_file, 'w', encoding='utf-8') as f:
            json.dump(strategy, f, indent=4, ensure_ascii=False)
        
        self.strategies[name] = strategy
    
    def get_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """
        دریافت استراتژی معاملاتی.
        
        Args:
            name: نام استراتژی
            
        Returns:
            اطلاعات استراتژی
        """
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """
        دریافت لیست استراتژی‌های معاملاتی.
        
        Returns:
            لیست استراتژی‌ها
        """
        return list(self.strategies.values())
    
    def execute_trade(self, symbol: str, strategy: str, 
                     position_type: str, size: float,
                     entry_price: float, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        اجرای معامله.
        
        Args:
            symbol: نماد
            strategy: نام استراتژی
            position_type: نوع موقعیت (LONG یا SHORT)
            size: حجم معامله
            entry_price: قیمت ورود
            parameters: پارامترهای معامله (اختیاری)
            
        Returns:
            اطلاعات معامله
        """
        if strategy not in self.strategies:
            raise ValueError(f"استراتژی {strategy} یافت نشد")
        
        trade = {
            'symbol': symbol,
            'strategy': strategy,
            'position_type': position_type,
            'size': size,
            'entry_price': entry_price,
            'entry_time': datetime.now().isoformat(),
            'status': 'OPEN',
            'parameters': parameters or {},
            'created_at': datetime.now().isoformat()
        }
        
        # ذخیره معامله
        trade_file = self.trades_dir / f"trade_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(trade_file, 'w', encoding='utf-8') as f:
            json.dump(trade, f, indent=4, ensure_ascii=False)
        
        return trade
    
    def close_trade(self, trade_id: str, exit_price: float) -> Dict[str, Any]:
        """
        بستن معامله.
        
        Args:
            trade_id: شناسه معامله
            exit_price: قیمت خروج
            
        Returns:
            اطلاعات معامله به‌روز شده
        """
        trade_file = self.trades_dir / f"{trade_id}.json"
        if not trade_file.exists():
            raise FileNotFoundError(f"معامله {trade_id} یافت نشد")
        
        with open(trade_file, 'r', encoding='utf-8') as f:
            trade = json.load(f)
        
        if trade['status'] != 'OPEN':
            raise ValueError(f"معامله {trade_id} قبلاً بسته شده است")
        
        # محاسبه سود/زیان
        if trade['position_type'] == 'LONG':
            pnl = (exit_price - trade['entry_price']) * trade['size']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['size']
        
        # به‌روزرسانی معامله
        trade.update({
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'pnl': pnl,
            'status': 'CLOSED'
        })
        
        with open(trade_file, 'w', encoding='utf-8') as f:
            json.dump(trade, f, indent=4, ensure_ascii=False)
        
        return trade
    
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        دریافت اطلاعات معامله.
        
        Args:
            trade_id: شناسه معامله
            
        Returns:
            اطلاعات معامله
        """
        trade_file = self.trades_dir / f"{trade_id}.json"
        if not trade_file.exists():
            return None
        
        with open(trade_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_trades(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        دریافت لیست معاملات.
        
        Args:
            status: وضعیت معامله (اختیاری)
            
        Returns:
            لیست معاملات
        """
        trades = []
        for trade_file in self.trades_dir.glob("trade_*.json"):
            with open(trade_file, 'r', encoding='utf-8') as f:
                trade = json.load(f)
                if status is None or trade['status'] == status:
                    trades.append(trade)
        
        return sorted(trades, key=lambda x: x['created_at'], reverse=True)
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        دریافت آمار معاملات.
        
        Returns:
            آمار معاملات
        """
        trades = self.list_trades()
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'average_pnl': 0.0,
                'max_pnl': 0.0,
                'min_pnl': 0.0
            }
        
        closed_trades = [t for t in trades if t['status'] == 'CLOSED']
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        pnls = [t['pnl'] for t in closed_trades]
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(closed_trades) - len(winning_trades),
            'total_pnl': total_pnl,
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0.0,
            'average_pnl': total_pnl / len(closed_trades) if closed_trades else 0.0,
            'max_pnl': max(pnls) if pnls else 0.0,
            'min_pnl': min(pnls) if pnls else 0.0
        }
    
    def get_strategy_performance(self, strategy: str) -> Dict[str, Any]:
        """
        دریافت عملکرد استراتژی.
        
        Args:
            strategy: نام استراتژی
            
        Returns:
            عملکرد استراتژی
        """
        trades = [t for t in self.list_trades() if t['strategy'] == strategy]
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'average_pnl': 0.0,
                'max_pnl': 0.0,
                'min_pnl': 0.0
            }
        
        closed_trades = [t for t in trades if t['status'] == 'CLOSED']
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        pnls = [t['pnl'] for t in closed_trades]
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(closed_trades) - len(winning_trades),
            'total_pnl': total_pnl,
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0.0,
            'average_pnl': total_pnl / len(closed_trades) if closed_trades else 0.0,
            'max_pnl': max(pnls) if pnls else 0.0,
            'min_pnl': min(pnls) if pnls else 0.0
        } 