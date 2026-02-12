import pytest
import pandas as pd
import os
from src.engine import BacktestEngine
from src.data_loader import BacktestConfig

class TestBacktestEngine:
    """测试回测引擎功能"""
    
    def setup_method(self):
        """设置测试环境"""
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=30),
            'open': [100.0] * 30,
            'high': [101.0] * 30,
            'low': [99.0] * 30,
            'close': [100.0, 101.0, 102.0, 101.5, 102.5] * 6,
            'volume': [1000] * 30
        })
        
        # 创建测试配置
        self.config = BacktestConfig(
            symbol_name='TEST',
            ticker='TEST',
            initial_capital=100000,
            start_date='2023-01-01',
            end_date='2023-01-30'
        )
    
    def test_engine_initialization(self):
        """测试引擎初始化"""
        engine = BacktestEngine(self.config, self.test_data)
        assert engine is not None
        assert engine.cfg.symbol_name == 'TEST'
        assert engine.cfg.initial_capital == 100000
    
    def test_strategy_config_loading(self):
        """测试策略配置加载"""
        engine = BacktestEngine(self.config, self.test_data)
        assert engine.trade_config is not None
        assert 'market_types' in engine.trade_config
    
    def test_invalid_config_path(self):
        """测试无效配置路径"""
        with pytest.raises(Exception):
            BacktestEngine(self.config, self.test_data, 'invalid_path.yaml')
