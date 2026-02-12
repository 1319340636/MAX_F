# ==========================================
# 文件名: src/mechanism.py
# 功能: 市场清算与战况播报 (纯观察者模式)
# ==========================================

import logging
from typing import List
from src.models import Prediction

logger = logging.getLogger(__name__)

class PredictionMarket:
    def __init__(self):
        pass

    def resolve_market(self, predictions: List[Prediction], outcome: str):
        """
        市场结算与战况播报
        注意：真正的权重调整已移交至 DecisionFusionEngine，此处仅负责日志输出。
        """
        # 1. 统计红黑榜
        winners = []
        losers = []
        observers = []

        for pred in predictions:
            if pred.direction == "HOLD":
                observers.append(pred.agent_id)
                continue
            
            if pred.direction == outcome:
                winners.append(f"{pred.agent_id}(Conf:{pred.confidence:.2f})")
            else:
                losers.append(f"{pred.agent_id}(Conf:{pred.confidence:.2f})")

        # 2. 生成战报日志
        log_msg = f"⚖️ [结算] 结果:{outcome} | "
        
        if winners:
            log_msg += f"✅ 赢家: {', '.join(winners)} | "
        else:
            log_msg += "✅ 赢家: 无 | "
            
        if losers:
            log_msg += f"❌ 输家: {', '.join(losers)}"
        else:
            log_msg += "❌ 输家: 无"
            
        # if observers:
        #     log_msg += f" | 😶 观望: {', '.join(observers)}"

        # 3. 输出统一格式的日志
        logger.info(log_msg)