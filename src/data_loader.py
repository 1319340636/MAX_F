# src/data_loader.py
import os
import time
import pandas as pd
import yfinance as yf
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    symbol_name: str
    ticker: str
    start_date: Optional[str] = None 
    end_date: Optional[str] = None
    min_kline_count: int = 40
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    initial_capital: float = 100.0
    max_workers: int = 10 
    
    @property
    def total_cost_rate(self) -> float:
        return self.commission_rate + self.slippage_rate

    @property
    def report_dir(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dir_path = f"logs/{timestamp}_{self.symbol_name.upper()}"
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

class DataManager:
    CACHE_DIR = "data_cache"
    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def get_data(self, config: BacktestConfig) -> Optional[pd.DataFrame]:
        s_date = config.start_date if config.start_date else "MAX"
        e_date = config.end_date if config.end_date else "MAX"
        cache_path = os.path.join(self.CACHE_DIR, f"{config.ticker}_{s_date}_{e_date}.csv")
        
        if os.path.exists(cache_path):
            logger.info(f"📥 加载缓存数据: {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            logger.info(f"🌐 下载数据 {config.ticker}...")
            df = self._download_safe(config)
            if not df.empty: df.to_csv(cache_path)

        df = self._process_indicators(df)
        if len(df) < config.min_kline_count: return None
        config.start_date, config.end_date = df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")
        logger.info(f"✅ 数据就绪: {config.start_date} ~ {config.end_date} ({len(df)}条)")
        return df

    def _download_safe(self, config):
        try:
            df = yf.download(config.ticker, start=config.start_date, end=config.end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df
        except: return pd.DataFrame()

    def _process_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.capitalize() for c in df.columns]
        df["MA60"] = df["Close"].rolling(60).mean()
        df["Volatility"] = df["Close"].pct_change().rolling(20).std()
        return df.dropna()