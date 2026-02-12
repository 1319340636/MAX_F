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
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            except Exception as e:
                logger.error(f"缓存文件读取失败: {e}")
                # 删除损坏的缓存文件
                os.remove(cache_path)
                # 重新下载数据
                logger.info("重新下载数据...")
                df = self._download_safe(config)
                if not df.empty:
                    df.to_csv(cache_path)
        else:
            logger.info(f"🌐 下载数据 {config.ticker}...")
            df = self._download_safe(config)
            if not df.empty:
                df.to_csv(cache_path)

        df = self._process_indicators(df)
        if len(df) < config.min_kline_count:
            return None
        config.start_date, config.end_date = df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")
        logger.info(f"✅ 数据就绪: {config.start_date} ~ {config.end_date} ({len(df)}条)")
        return df

    def _download_safe(self, config):
        """安全下载数据，添加超时和重试机制"""
        max_retries = 3
        timeout = 30  # 30秒超时
        
        for attempt in range(max_retries):
            try:
                logger.info(f"尝试下载数据 (尝试 {attempt+1}/{max_retries}): {config.ticker}")
                logger.info(f"时间范围: {config.start_date} 到 {config.end_date}")
                
                # 添加超时设置
                df = yf.download(
                    config.ticker, 
                    start=config.start_date, 
                    end=config.end_date, 
                    progress=False
                )
                
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                
                logger.info(f"数据下载成功，共 {len(df)} 条记录")
                return df
            except Exception as e:
                logger.error(f"下载失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 3
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error("所有尝试都失败了，返回空DataFrame")
                    return pd.DataFrame()

    def _process_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据指标，添加错误处理"""
        try:
            logger.info(f"开始处理数据指标，原始数据行数: {len(df)}")
            
            # 转换列名
            df.columns = [c.capitalize() for c in df.columns]
            logger.info(f"列名转换完成，列名: {list(df.columns)}")
            
            # 计算MA60
            if 'Close' in df.columns:
                df["MA60"] = df["Close"].rolling(60).mean()
                logger.info("MA60计算完成")
                
                # 计算波动率
                df["Volatility"] = df["Close"].pct_change().rolling(20).std()
                logger.info("波动率计算完成")
                
                # 移除NA值
                df_clean = df.dropna()
                logger.info(f"数据处理完成，处理后数据行数: {len(df_clean)}")
                return df_clean
            else:
                logger.error("数据中缺少 'Close' 列")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()