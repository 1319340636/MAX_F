# ==========================================
# 文件名: src/utils.py
# ==========================================

import os
import yaml
import logging  # ✅ 修复了这里：补上了 logging
import asyncio
import random
from dataclasses import dataclass
from typing import Dict, Optional, Any, Callable
from pydantic import BaseModel, Field, ValidationError

# 尝试导入 Prediction 模型，用于 async_retry 的兜底返回
# 如果 src.models 还没定义，可能需要检查文件结构
try:
    from src.models import Prediction
except ImportError:
    # 临时定义一个占位符，防止循环引用报错
    @dataclass
    class Prediction:
        agent_id: str
        direction: str
        confidence: float
        stake: float
        reasoning: str

# 配置日志记录器
logger = logging.getLogger(__name__)

# ===================== 0. 动态加载 YAML 配置 =====================
def load_vol_config_from_yaml(yaml_path="config/trade_config.yaml"):
    """
    直接读取 YAML 文件，避免 import config 导致的循环引用。
    如果读取失败，则回退到默认硬编码值。
    """
    default_config = {
        "gold": {"trend_th": 0.008, "extreme_th": 0.018, "atr_stop_mult": 2.0},
        "btc":  {"trend_th": 0.02,  "extreme_th": 0.04,  "atr_stop_mult": 3.0},
        "default": {"trend_th": 0.01, "extreme_th": 0.03, "atr_stop_mult": 2.0}
    }

    if not os.path.exists(yaml_path):
        # logger.warning(f"⚠️ Utils 警告: 找不到 {yaml_path}，使用默认参数。")
        return default_config

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            raw_cfg = yaml.safe_load(f)
        
        gold_cfg = raw_cfg.get("market_types", {}).get("gold", {})
        gold_flashbulb = gold_cfg.get("flashbulb_vol", 0.018)
        
        crypto_cfg = raw_cfg.get("market_types", {}).get("crypto", {})
        crypto_flashbulb = crypto_cfg.get("flashbulb_vol", 0.04)

        return {
            "gold": {
                "trend_th": 0.008,  
                "extreme_th": gold_flashbulb,
                "atr_stop_mult": 2.0
            },
            "btc": {
                "trend_th": 0.02,
                "extreme_th": crypto_flashbulb,
                "atr_stop_mult": 3.0
            },
            "default": default_config["default"]
        }
    except Exception as e:
        logger.warning(f"⚠️ YAML 解析失败: {e}，使用默认参数。")
        return default_config

# 程序启动时自动执行加载
VOL_CONFIG = load_vol_config_from_yaml()

# ===================== 1. 异步重试装饰器 (关键修复) =====================
def async_retry(retries: int = 3, delay: float = 1.0):
    """
    异步函数的重试装饰器。
    用于 LLM API 调用等不稳定网络操作。
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # 指数退避 + 随机抖动
                    sleep_time = delay * (2 ** i) + random.uniform(0, 1)
                    await asyncio.sleep(sleep_time)
            
            logger.error(f"❌ 异步重试 {retries} 次后失败: {last_exception}")
            # 失败兜底返回 HOLD，防止程序崩溃
            return Prediction(
                agent_id="Unknown", 
                direction="HOLD", 
                confidence=0.0, 
                stake=0.0, 
                reasoning=f"Timeout/Error after retries: {str(last_exception)[:100]}"
            )
        return wrapper
    return decorator

# ===================== 2. 数据验证层 =====================
class RawMarketData(BaseModel):
    Close: float = Field(..., description="收盘价")
    High: float = Field(..., gt=0)
    Low: float = Field(..., gt=0)
    Volatility: float = Field(default=0.0) 
    MA5: float = Field(default=0.0)
    MA20: float = Field(default=0.0)
    MA60: float = Field(default=0.0)
    
    class Config:
        extra = "ignore" 

# ===================== 3. 上下文模型 =====================
@dataclass
class MarketContext:
    timestamp: Any
    symbol: str
    price: float
    volatility: float
    period: str
    market_env: str
    news_summary: str
    support_level: float
    resistance_level: float
    bias_ma20: float
    
    # 交易状态
    position_size: float = 0.0
    avg_cost: float = 0.0

# ===================== 4. 辅助函数 =====================
def map_symbol_to_asset(symbol: str) -> str:
    s = symbol.upper()
    if any(x in s for x in ["XAU", "GC=F", "GOLD"]): return "gold"
    if any(x in s for x in ["BTC", "ETH"]): return "btc"
    return "default"

def get_market_env(vol: float, asset_type: str, bias: float) -> str:
    cfg = VOL_CONFIG.get(asset_type, VOL_CONFIG["default"])
    
    if vol > cfg["extreme_th"]: 
        vol_state = "【极端风暴】(Risk Manager 请高度警惕)"
    elif vol > cfg["trend_th"]: 
        vol_state = "【活跃趋势】(Tech Analyst 请寻找机会)"
    else: 
        vol_state = "【低波死鱼】(建议观望或区间操作)"
    
    bias_state = ""
    if bias > 0.035: bias_state = " + ⚠️严重超买"
    elif bias < -0.035: bias_state = " + ⚠️严重超卖"
    
    return f"{vol_state}{bias_state}"

# ===================== 5. 核心构建逻辑 =====================
def build_market_context(
    curr_row: Dict,
    curr_date: Any,
    real_symbol: str,
    period: str = "1D"
) -> Optional[MarketContext]:
    
    try:
        clean_data = RawMarketData(**curr_row)
    except ValidationError:
        return None

    data = clean_data
    asset_type = map_symbol_to_asset(real_symbol)

    pivot = (data.High + data.Low + data.Close) / 3
    r1 = 2 * pivot - data.Low
    s1 = 2 * pivot - data.High
    
    bias_pct = (data.Close - data.MA20) / data.MA20 if data.MA20 > 0 else 0.0

    if data.MA5 > data.MA20 > data.MA60:
        ma_state = "多头排列 (强势上涨结构)"
    elif data.MA5 < data.MA20 < data.MA60:
        ma_state = "空头排列 (弱势下跌结构)"
    else:
        ma_state = "均线纠缠 (方向不明)"

    pos_state = "站稳 MA20 上方" if data.Close > data.MA20 else "跌破 MA20 压制"
    
    news_summary = (
        f"【技术结构】{ma_state}，价格目前{pos_state}。\n"
        f"【乖离状态】MA20乖离率 {bias_pct:.2%}。\n"
        f"【关键点位】日内支撑(S1): {s1:.2f} / 阻力(R1): {r1:.2f}。"
    )

    market_env = get_market_env(data.Volatility, asset_type, bias_pct)

    return MarketContext(
        timestamp=curr_date,
        symbol=real_symbol,
        price=data.Close,
        volatility=data.Volatility,
        news_summary=news_summary,
        period=period,
        market_env=market_env,
        support_level=s1,
        resistance_level=r1,
        bias_ma20=bias_pct
    )