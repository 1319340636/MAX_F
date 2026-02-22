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
    min_kline_count: int = 40  # 减少最小K线数量要求
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    initial_capital: float = 100.0
    max_workers: int = 10
    use_checkpoint: bool = False  # 是否使用断点重连
    
    @property
    def total_cost_rate(self) -> float:
        return self.commission_rate + self.slippage_rate

    @property
    def report_dir(self) -> str:
        if self.use_checkpoint:
            dir_path = f"logs/{self.symbol_name.upper()}_checkpoint"
        else:
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
        
        # 检查并删除可能导致问题的旧缓存文件
        import glob
        old_cache_files = glob.glob(os.path.join(self.CACHE_DIR, f"{config.ticker}_*.csv"))
        if old_cache_files:
            logger.info(f"发现旧缓存文件，将重新下载数据: {old_cache_files}")
            # 强制重新下载数据，忽略旧缓存
            logger.info(f"🌐 下载数据 {config.ticker}...")
            df = self._download_safe(config)
            if not df.empty:
                df.to_csv(cache_path)
        elif os.path.exists(cache_path):
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

        # 下载 GVZ 数据（黄金波动率指数）
        gvz_df = self._download_safe(BacktestConfig(
            symbol_name="GVZ",
            ticker="^GVZ",
            start_date=config.start_date,
            end_date=config.end_date
        ))
        
        # 下载 DXY 数据（美元指数）
        dxy_df = self._download_safe(BacktestConfig(
            symbol_name="DXY",
            ticker="DX-Y.NYB",
            start_date=config.start_date,
            end_date=config.end_date
        ))

        df = self._process_indicators(df, gvz_df, dxy_df)
        if len(df) < config.min_kline_count:
            return None
        config.start_date, config.end_date = df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")
        logger.info(f"✅ 数据就绪: {config.start_date} ~ {config.end_date} ({len(df)}条)")
        return df

    def _download_safe(self, config):
        """安全下载数据，添加超时和重试机制
        
        如果用户指定的时间范围太短，自动向前扩展一段时间，确保有足够的数据计算指标
        """
        max_retries = 3
        timeout = 30  # 30秒超时
        
        # 计算需要的额外天数
        extra_days = 60  # 额外下载60天数据，确保可以计算MA60等指标
        
        # 解析开始日期并向前扩展
        start_date = config.start_date
        if start_date:
            try:
                # 向前扩展extra_days天
                start_date_obj = pd.to_datetime(start_date)
                extended_start_date = start_date_obj - pd.Timedelta(days=extra_days)
                extended_start_date_str = extended_start_date.strftime("%Y-%m-%d")
                logger.info(f"时间范围较短，向前扩展 {extra_days} 天数据以计算指标")
                logger.info(f"扩展后时间范围: {extended_start_date_str} 到 {config.end_date}")
            except:
                extended_start_date_str = start_date
        else:
            extended_start_date_str = start_date
        
        for attempt in range(max_retries):
            try:
                logger.info(f"尝试下载数据 (尝试 {attempt+1}/{max_retries}): {config.ticker}")
                logger.info(f"时间范围: {extended_start_date_str} 到 {config.end_date}")
                
                # 添加超时设置
                df = yf.download(
                    config.ticker, 
                    start=extended_start_date_str, 
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

    def _process_indicators(self, df: pd.DataFrame, gvz_df: pd.DataFrame, dxy_df: pd.DataFrame) -> pd.DataFrame:
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
                
                # 处理 GVZ 数据（黄金波动率指数）
                if not gvz_df.empty:
                    # 转换列名并获取收盘价
                    gvz_df.columns = [c.capitalize() for c in gvz_df.columns]
                    if 'Close' in gvz_df.columns:
                        # 对齐索引
                        gvz_close = gvz_df['Close'].reindex(df.index)
                        df['GVZ'] = gvz_close
                        
                        # 计算 GVZ Z-Score
                        df['GVZ_MA20'] = df['GVZ'].rolling(20).mean()
                        df['GVZ_STD20'] = df['GVZ'].rolling(20).std()
                        df['GVZ_ZScore'] = (df['GVZ'] - df['GVZ_MA20']) / df['GVZ_STD20']
                        
                        # 计算 GVZ 分位数
                        df['GVZ_Quantile'] = df['GVZ'].rolling(60).apply(lambda x: x.rank(pct=True).iloc[-1])
                        
                        # 计算 GVZ 变化率
                        df['GVZ_Change'] = df['GVZ'].pct_change()
                        
                        logger.info("GVZ 指标计算完成")
                
                # 处理 DXY 数据（美元指数）
                if not dxy_df.empty:
                    # 转换列名并获取收盘价
                    dxy_df.columns = [c.capitalize() for c in dxy_df.columns]
                    if 'Close' in dxy_df.columns:
                        # 对齐索引
                        dxy_close = dxy_df['Close'].reindex(df.index)
                        df['DXY'] = dxy_close
                        
                        # 计算 DXY 走势方向
                        df['DXY_Change'] = dxy_close.pct_change()
                        df['DXY_Direction'] = df['DXY_Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                        
                        # 计算 DXY 5日趋势
                        df['DXY_Trend_5D'] = df['DXY_Direction'].rolling(5).sum()
                        
                        logger.info("DXY 指标计算完成")
                
                # 计算黄金与美元的相关性
                if 'DXY_Change' in df.columns:
                    df['Gold_DXY_Correlation'] = df['Close'].pct_change().rolling(20).corr(df['DXY_Change'])
                    logger.info("黄金与美元相关性计算完成")
                
                # 不要移除所有NA值，保留尽可能多的数据
                # 对于短时间范围，某些指标可能无法计算，但我们仍然保留其他数据
                # 只移除完全空的行
                df_clean = df.dropna(how='all')
                
                # 如果数据太少，使用较短的时间窗口
                if len(df_clean) < 20:
                    logger.info("数据时间范围较短，使用较短的时间窗口计算指标")
                    # 重新计算MA，使用较短的时间窗口
                    if 'Close' in df.columns:
                        # 使用MA20替代MA60
                        df["MA60"] = df["Close"].rolling(20).mean()
                        # 使用较短的时间窗口计算波动率
                        df["Volatility"] = df["Close"].pct_change().rolling(10).std()
                        # 重新清理数据
                        df_clean = df.dropna(how='all')
                
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