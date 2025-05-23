"""ماژول مدیریت پایگاه داده.

این ماژول مسئول مدیریت اتصال به پایگاه داده و عملیات‌های مربوط به آن است.
"""

import os
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime


Base = declarative_base()


class MarketData(Base):
    """جدول داده‌های بازار."""
    
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    indicators = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patterns = relationship("Pattern", back_populates="market_data")


class Pattern(Base):
    """جدول الگوهای شناسایی شده."""
    
    __tablename__ = 'patterns'
    
    id = Column(Integer, primary_key=True)
    market_data_id = Column(Integer, ForeignKey('market_data.id'), nullable=False)
    pattern_type = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    market_data = relationship("MarketData", back_populates="patterns")
    trades = relationship("Trade", back_populates="pattern")


class Trade(Base):
    """جدول معاملات انجام شده."""
    
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    pattern_id = Column(Integer, ForeignKey('patterns.id'), nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    position_type = Column(String(10), nullable=False)  # LONG or SHORT
    size = Column(Float, nullable=False)
    pnl = Column(Float)
    status = Column(String(20), nullable=False)  # OPEN, CLOSED, CANCELLED
    parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    pattern = relationship("Pattern", back_populates="trades")


class DatabaseManager:
    """مدیریت پایگاه داده."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه مدیر پایگاه داده.
        
        Args:
            config: تنظیمات پایگاه داده
        """
        self.config = config
        self.engine = create_engine(
            config['url'],
            pool_size=config['pool_size'],
            pool_recycle=config['pool_recycle']
        )
        self.Session = sessionmaker(bind=self.engine)
        
        # ایجاد جداول
        Base.metadata.create_all(self.engine)
    
    def add_market_data(self, data: Dict[str, Any]) -> MarketData:
        """
        افزودن داده‌های بازار.
        
        Args:
            data: داده‌های بازار
            
        Returns:
            رکورد داده‌های بازار
        """
        session = self.Session()
        try:
            market_data = MarketData(**data)
            session.add(market_data)
            session.commit()
            return market_data
        finally:
            session.close()
    
    def add_pattern(self, pattern: Dict[str, Any]) -> Pattern:
        """
        افزودن الگوی شناسایی شده.
        
        Args:
            pattern: اطلاعات الگو
            
        Returns:
            رکورد الگو
        """
        session = self.Session()
        try:
            pattern_obj = Pattern(**pattern)
            session.add(pattern_obj)
            session.commit()
            return pattern_obj
        finally:
            session.close()
    
    def add_trade(self, trade: Dict[str, Any]) -> Trade:
        """
        افزودن معامله.
        
        Args:
            trade: اطلاعات معامله
            
        Returns:
            رکورد معامله
        """
        session = self.Session()
        try:
            trade_obj = Trade(**trade)
            session.add(trade_obj)
            session.commit()
            return trade_obj
        finally:
            session.close()
    
    def get_market_data(self, symbol: str, 
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[MarketData]:
        """
        دریافت داده‌های بازار.
        
        Args:
            symbol: نماد
            start_time: زمان شروع (اختیاری)
            end_time: زمان پایان (اختیاری)
            
        Returns:
            لیست داده‌های بازار
        """
        session = self.Session()
        try:
            query = session.query(MarketData).filter(MarketData.symbol == symbol)
            
            if start_time:
                query = query.filter(MarketData.timestamp >= start_time)
            if end_time:
                query = query.filter(MarketData.timestamp <= end_time)
            
            return query.order_by(MarketData.timestamp).all()
        finally:
            session.close()
    
    def get_patterns(self, pattern_type: Optional[str] = None,
                    min_confidence: Optional[float] = None) -> List[Pattern]:
        """
        دریافت الگوهای شناسایی شده.
        
        Args:
            pattern_type: نوع الگو (اختیاری)
            min_confidence: حداقل اطمینان (اختیاری)
            
        Returns:
            لیست الگوها
        """
        session = self.Session()
        try:
            query = session.query(Pattern)
            
            if pattern_type:
                query = query.filter(Pattern.pattern_type == pattern_type)
            if min_confidence:
                query = query.filter(Pattern.confidence >= min_confidence)
            
            return query.order_by(Pattern.created_at.desc()).all()
        finally:
            session.close()
    
    def get_trades(self, status: Optional[str] = None) -> List[Trade]:
        """
        دریافت معاملات.
        
        Args:
            status: وضعیت معامله (اختیاری)
            
        Returns:
            لیست معاملات
        """
        session = self.Session()
        try:
            query = session.query(Trade)
            
            if status:
                query = query.filter(Trade.status == status)
            
            return query.order_by(Trade.created_at.desc()).all()
        finally:
            session.close()
    
    def update_trade(self, trade_id: int, updates: Dict[str, Any]) -> Optional[Trade]:
        """
        به‌روزرسانی معامله.
        
        Args:
            trade_id: شناسه معامله
            updates: اطلاعات به‌روزرسانی
            
        Returns:
            رکورد معامله به‌روز شده
        """
        session = self.Session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                for key, value in updates.items():
                    setattr(trade, key, value)
                session.commit()
                return trade
            return None
        finally:
            session.close()
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        دریافت آمار معاملات.
        
        Returns:
            آمار معاملات
        """
        session = self.Session()
        try:
            total_trades = session.query(Trade).count()
            closed_trades = session.query(Trade).filter(Trade.status == 'CLOSED').count()
            open_trades = session.query(Trade).filter(Trade.status == 'OPEN').count()
            
            total_pnl = session.query(func.sum(Trade.pnl)).filter(
                Trade.status == 'CLOSED'
            ).scalar() or 0.0
            
            winning_trades = session.query(Trade).filter(
                Trade.status == 'CLOSED',
                Trade.pnl > 0
            ).count()
            
            return {
                'total_trades': total_trades,
                'closed_trades': closed_trades,
                'open_trades': open_trades,
                'total_pnl': total_pnl,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / closed_trades if closed_trades > 0 else 0.0
            }
        finally:
            session.close() 