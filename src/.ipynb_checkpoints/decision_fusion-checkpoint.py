# ==========================================
# 文件名: src/decision_fusion.py
# 功能: 动态加权决策引擎 (Ensemble Learning)
# ==========================================

from collections import deque
from typing import List, Dict, Any

class AgentPerformance:
    """追踪单个 Agent 的近期战绩"""
    def __init__(self, agent_id: str, window_size=50):
        self.agent_id = agent_id
        # 滑动窗口：只看最近 50 次表现，适应市场风格切换
        self.history = deque(maxlen=window_size)
        self.current_weight = 1.0  # 初始权重

    def update(self, is_correct: bool):
        # 记录：1=猜对，0=猜错
        self.history.append(1 if is_correct else 0)
        self._recalculate_weight()

    def _recalculate_weight(self):
        if not self.history:
            return
        
        # 计算近期胜率
        win_rate = sum(self.history) / len(self.history)
        
        # 💡 动态权重公式：
        # 胜率 50% -> 权重 1.0
        # 胜率 80% -> 权重 1.6 (话语权大增)
        # 胜率 20% -> 权重 0.4 (基本被忽略)
        self.current_weight = max(0.2, win_rate * 2.0)
        
        # 连胜奖励：最近 3 次全对，权重额外 +20%
        if len(self.history) >= 3 and sum(list(self.history)[-3:]) == 3:
            self.current_weight *= 1.2

class DecisionFusionEngine:
    """决策融合主引擎"""
    def __init__(self, agent_ids: List[str]):
        # 为每个 Agent 初始化战绩追踪器
        self.trackers = {aid: AgentPerformance(aid) for aid in agent_ids}

    def update_performance(self, actions: List[Any], outcome: str):
        """每局结束后调用，更新战绩"""
        for act in actions:
            # HOLD 不计入胜率统计，避免 Agent 靠“装死”刷分
            if act.direction == "HOLD": 
                continue
            
            # 判断是否与实际结果一致
            is_correct = (act.direction == outcome)
            self.trackers[act.agent_id].update(is_correct)

    def get_weighted_decision(self, actions: List[Any]) -> Dict:
        """核心：加权投票"""
        votes = {"LONG": 0.0, "SHORT": 0.0}
        total_power = 0.0
        agent_weights_snapshot = {}

        for act in actions:
            if act.direction == "HOLD": 
                continue
            
            # 🔥 核心：实际投票权 = 信心 * 历史权重
            base_weight = self.trackers[act.agent_id].current_weight
            final_power = act.confidence * base_weight
            
            agent_weights_snapshot[act.agent_id] = round(final_power, 2)

            if act.direction in votes:
                votes[act.direction] += final_power
                total_power += final_power

        # 决策生成逻辑
        final_dir = "HOLD"
        final_stake = 0.0
        
        # 阈值判断：多空力量需有明显差距 (1.2倍)
        if votes["LONG"] > votes["SHORT"] * 1.2 and votes["LONG"] > 0.5:
            final_dir = "LONG"
            # 信心归一化
            final_stake = min(votes["LONG"] / (total_power + 0.1), 1.0)
            
        elif votes["SHORT"] > votes["LONG"] * 1.2 and votes["SHORT"] > 0.5:
            final_dir = "SHORT"
            final_stake = min(votes["SHORT"] / (total_power + 0.1), 1.0)

        return {
            "direction": final_dir,
            "stake": final_stake,
            "weights": agent_weights_snapshot  # 返回权重快照用于打印日志
        }