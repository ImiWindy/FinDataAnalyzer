"""ماژول مدیریت ریسک.

این ماژول مسئول مدیریت ریسک و محاسبه معیارهای ریسک است.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats


class RiskManager:
    """مدیریت ریسک."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه مدیر ریسک.
        
        Args:
            config: تنظیمات مدیریت ریسک
        """
        self.config = config
        self.risk_limits = config['risk_limits']
        self.position_limits = config['position_limits']
        self.stop_loss_limits = config['stop_loss_limits']
    
    def calculate_position_size(self, capital: float, risk_per_trade: float,
                              entry_price: float, stop_loss: float) -> float:
        """
        محاسبه حجم معامله.
        
        Args:
            capital: سرمایه
            risk_per_trade: ریسک به ازای هر معامله
            entry_price: قیمت ورود
            stop_loss: حد ضرر
            
        Returns:
            حجم معامله
        """
        risk_amount = capital * risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            raise ValueError("قیمت ورود و حد ضرر نمی‌توانند یکسان باشند")
        
        position_size = risk_amount / price_risk
        
        # اعمال محدودیت‌های حجم معامله
        position_size = min(position_size, self.position_limits['max_size'])
        position_size = max(position_size, self.position_limits['min_size'])
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, position_type: str,
                          atr: float, atr_multiplier: float = 2.0) -> float:
        """
        محاسبه حد ضرر.
        
        Args:
            entry_price: قیمت ورود
            position_type: نوع موقعیت (LONG یا SHORT)
            atr: میانگین محدوده واقعی
            atr_multiplier: ضریب ATR
            
        Returns:
            حد ضرر
        """
        stop_distance = atr * atr_multiplier
        
        if position_type == 'LONG':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        # اعمال محدودیت‌های حد ضرر
        if position_type == 'LONG':
            max_loss = entry_price * (1 - self.stop_loss_limits['max_percentage'])
            stop_loss = max(stop_loss, max_loss)
        else:
            max_loss = entry_price * (1 + self.stop_loss_limits['max_percentage'])
            stop_loss = min(stop_loss, max_loss)
        
        return stop_loss
    
    def calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        محاسبه معیارهای ریسک.
        
        Args:
            returns: بازدهی‌ها
            
        Returns:
            معیارهای ریسک
        """
        if len(returns) < 2:
            raise ValueError("برای محاسبه معیارهای ریسک حداقل به 2 داده نیاز است")
        
        # محاسبه معیارهای پایه
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # محاسبه معیارهای ریسک
        metrics = {
            'volatility': std_return * np.sqrt(252),  # نوسان‌پذیری سالانه
            'sharpe_ratio': (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0,
            'sortino_ratio': (mean_return * 252) / (np.std(returns[returns < 0], ddof=1) * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'var_95': self._calculate_var(returns, 0.95),
            'cvar_95': self._calculate_cvar(returns, 0.95),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        محاسبه حداکثر افت.
        
        Args:
            returns: بازدهی‌ها
            
        Returns:
            حداکثر افت
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        return abs(drawdowns.min())
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        محاسبه ارزش در معرض ریسک.
        
        Args:
            returns: بازدهی‌ها
            confidence_level: سطح اطمینان
            
        Returns:
            ارزش در معرض ریسک
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        محاسبه ارزش در معرض ریسک شرطی.
        
        Args:
            returns: بازدهی‌ها
            confidence_level: سطح اطمینان
            
        Returns:
            ارزش در معرض ریسک شرطی
        """
        var = self._calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def check_risk_limits(self, portfolio_value: float, 
                         open_positions: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        بررسی محدودیت‌های ریسک.
        
        Args:
            portfolio_value: ارزش پرتفوی
            open_positions: موقعیت‌های باز
            
        Returns:
            وضعیت محدودیت‌های ریسک
        """
        # محاسبه ریسک کل
        total_risk = sum(pos['size'] * abs(pos['entry_price'] - pos['stop_loss'])
                        for pos in open_positions)
        risk_percentage = total_risk / portfolio_value
        
        # بررسی محدودیت‌های ریسک
        risk_checks = {
            'max_portfolio_risk': risk_percentage <= self.risk_limits['max_portfolio_risk'],
            'max_position_risk': all(pos['size'] * abs(pos['entry_price'] - pos['stop_loss']) / portfolio_value
                                   <= self.risk_limits['max_position_risk']
                                   for pos in open_positions),
            'max_open_positions': len(open_positions) <= self.risk_limits['max_open_positions'],
            'max_correlation': self._check_correlation_limit(open_positions)
        }
        
        return risk_checks
    
    def _check_correlation_limit(self, positions: List[Dict[str, Any]]) -> bool:
        """
        بررسی محدودیت همبستگی.
        
        Args:
            positions: موقعیت‌ها
            
        Returns:
            وضعیت محدودیت همبستگی
        """
        if len(positions) < 2:
            return True
        
        # محاسبه همبستگی بین موقعیت‌ها
        returns = []
        for pos in positions:
            if 'returns' in pos:
                returns.append(pos['returns'])
        
        if not returns:
            return True
        
        returns_array = np.array(returns)
        correlation_matrix = np.corrcoef(returns_array)
        
        # بررسی همبستگی‌های بالا
        high_correlations = np.abs(correlation_matrix) > self.risk_limits['max_correlation']
        np.fill_diagonal(high_correlations, False)
        
        return not np.any(high_correlations)
    
    def calculate_portfolio_metrics(self, portfolio_value: float,
                                  positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        محاسبه معیارهای پرتفوی.
        
        Args:
            portfolio_value: ارزش پرتفوی
            positions: موقعیت‌ها
            
        Returns:
            معیارهای پرتفوی
        """
        if not positions:
            return {
                'total_value': portfolio_value,
                'total_risk': 0.0,
                'diversification_score': 1.0,
                'position_weights': {},
                'risk_weights': {}
            }
        
        # محاسبه وزن‌های موقعیت
        position_values = {pos['symbol']: pos['size'] * pos['entry_price']
                         for pos in positions}
        total_position_value = sum(position_values.values())
        
        position_weights = {symbol: value / total_position_value
                          for symbol, value in position_values.items()}
        
        # محاسبه وزن‌های ریسک
        risk_values = {pos['symbol']: pos['size'] * abs(pos['entry_price'] - pos['stop_loss'])
                      for pos in positions}
        total_risk = sum(risk_values.values())
        
        risk_weights = {symbol: value / total_risk
                       for symbol, value in risk_values.items()}
        
        # محاسبه امتیاز تنوع‌بخشی
        if len(positions) > 1:
            weights = np.array(list(position_weights.values()))
            diversification_score = 1 - np.sum(weights ** 2)
        else:
            diversification_score = 0.0
        
        return {
            'total_value': portfolio_value,
            'total_risk': total_risk,
            'diversification_score': diversification_score,
            'position_weights': position_weights,
            'risk_weights': risk_weights
        } 