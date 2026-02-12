# src/models.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentState:
    agent_id: str
    role: str
    balance: float = 10000.0

@dataclass
class MarketContext:
    timestamp: any
    symbol: str
    price: float
    volatility: float
    news_summary: str
    
    # ✅ 新增：实战交易上下文
    period: str = "1D"           # 周期 (1H/4H/1D)
    market_env: str = "震荡市"    # 市场环境 (趋势/震荡/极端)
    position_size: float = 0.0   # 当前持仓 (0为无仓位)
    avg_cost: float = 0.0        # 持仓成本

@dataclass
class Prediction:
    agent_id: str
    direction: str       # LONG/SHORT/HOLD
    confidence: float    # 0.0 - 1.0
    stake: float         # 建议仓位
    reasoning: str
    
    # ✅ 新增：可执行的交易参数
    stop_loss: float = 0.0      # 止损价
    take_profit: float = 0.0    # 止盈价
    position_ratio: float = 0.0 # 建议仓位比例 (0.0 - 1.0)