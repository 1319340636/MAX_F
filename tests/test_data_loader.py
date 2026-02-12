import pytest
import pandas as pd
from src.data_loader import DataManager, BacktestConfig

class TestDataManager:
    """测试数据加载模块功能"""
    
    def test_backtest_config_creation(self):
        """测试回测配置创建"""
        config = BacktestConfig(
            symbol_name='GOLD',
            ticker='GC=F',
            initial_capital=100000,
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        assert config.symbol_name == 'GOLD'
        assert config.initial_capital == 100000
        assert config.start_date == '2023-01-01'
        assert config.end_date == '2023-01-31'
    
    def test_data_manager_initialization(self):
        """测试数据管理器初始化"""
        manager = DataManager()
        assert manager is not None
    
    def test_invalid_config_parameters(self):
        """测试无效配置参数"""
        with pytest.raises(Exception):
            BacktestConfig(
                asset_name='',
                initial_balance=-1000,
                start_date='2023-01-01',
                end_date='2022-01-01'  # 结束日期早于开始日期
            )
