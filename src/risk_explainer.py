# ==========================================
# 文件名: src/risk_explainer.py
# 功能: 风险控制可解释性模块
# ==========================================

import json
import os
from typing import Dict, List, Any

class RiskExplainer:
    """风险控制可解释性模块"""
    
    def __init__(self):
        self.explanations = []
    
    def explain_decision(self, decision: Dict, agent_votes: List[Any], market_context: Dict) -> Dict:
        """解释决策过程"""
        explanation = {
            "decision": decision,
            "agent_analysis": self._analyze_agent_votes(agent_votes),
            "risk_assessment": self._assess_risk(decision, market_context),
            "constraint_analysis": self._analyze_constraints(decision),
            "confidence_breakdown": self._breakdown_confidence(agent_votes),
            "rationale": self._generate_rationale(decision, agent_votes, market_context)
        }
        
        self.explanations.append(explanation)
        return explanation
    
    def _analyze_agent_votes(self, agent_votes: List[Any]) -> Dict:
        """分析Agent投票"""
        analysis = {}
        for vote in agent_votes:
            if hasattr(vote, 'agent_id') and hasattr(vote, 'direction') and hasattr(vote, 'confidence'):
                analysis[vote.agent_id] = {
                    "direction": vote.direction,
                    "confidence": vote.confidence,
                    "reason": getattr(vote, 'reason', "No reasoning provided")
                }
        return analysis
    
    def _assess_risk(self, decision: Dict, market_context: Dict) -> Dict:
        """评估风险"""
        risk_level = "Low"
        risk_factors = []
        
        # 基于仓位评估风险
        stake = decision.get("stake", 0.0)
        if stake > 0.8:
            risk_level = "High"
            risk_factors.append("High position size")
        elif stake > 0.5:
            risk_level = "Medium"
            risk_factors.append("Medium position size")
        
        # 基于市场环境评估风险
        if market_context and "volatility" in market_context:
            volatility = market_context["volatility"]
            if volatility > 0.02:
                risk_level = "High"
                risk_factors.append("High market volatility")
            elif volatility > 0.01:
                risk_factors.append("Medium market volatility")
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "mitigation_strategies": self._generate_mitigation_strategies(risk_factors)
        }
    
    def _analyze_constraints(self, decision: Dict) -> Dict:
        """分析约束条件"""
        constraints = {}
        direction = decision.get("direction", "HOLD")
        stake = decision.get("stake", 0.0)
        
        # 硬约束检查
        constraints["hard_constraints"] = {
            "position_limit": stake <= 1.0,
            "direction_valid": direction in ["LONG", "SHORT", "HOLD"]
        }
        
        # 软约束评估
        constraints["soft_constraints"] = {
            "reasonable_stake": 0.1 <= stake <= 0.9 if direction != "HOLD" else True,
            "clear_direction": direction != "HOLD" or stake == 0.0
        }
        
        return constraints
    
    def _breakdown_confidence(self, agent_votes: List[Any]) -> Dict:
        """分解信心度"""
        confidence_breakdown = {
            "by_agent": {},
            "average_confidence": 0.0,
            "confidence_distribution": {
                "high": 0,  # > 0.7
                "medium": 0,  # 0.4 - 0.7
                "low": 0  # < 0.4
            }
        }
        
        total_confidence = 0.0
        valid_votes = 0
        
        for vote in agent_votes:
            if hasattr(vote, 'agent_id') and hasattr(vote, 'confidence'):
                confidence = vote.confidence
                confidence_breakdown["by_agent"][vote.agent_id] = confidence
                total_confidence += confidence
                valid_votes += 1
                
                # 分类信心度
                if confidence > 0.7:
                    confidence_breakdown["confidence_distribution"]["high"] += 1
                elif confidence > 0.4:
                    confidence_breakdown["confidence_distribution"]["medium"] += 1
                else:
                    confidence_breakdown["confidence_distribution"]["low"] += 1
        
        if valid_votes > 0:
            confidence_breakdown["average_confidence"] = total_confidence / valid_votes
        
        return confidence_breakdown
    
    def _generate_rationale(self, decision: Dict, agent_votes: List[Any], market_context: Dict) -> str:
        """生成决策理由"""
        direction = decision.get("direction", "HOLD")
        stake = decision.get("stake", 0.0)
        
        # 分析Agent共识
        agent_directions = []
        for vote in agent_votes:
            if hasattr(vote, 'direction') and vote.direction != "HOLD":
                agent_directions.append(vote.direction)
        
        # 生成理由
        rationale_parts = []
        
        if direction == "HOLD":
            rationale_parts.append("系统决定保持观望，因为")
            if not agent_directions:
                rationale_parts.append("所有Agent都建议观望")
            else:
                rationale_parts.append("Agent之间存在分歧，无法形成明确共识")
        else:
            rationale_parts.append(f"系统决定{direction}，仓位大小{stake:.2f}，因为")
            
            # 分析共识
            if agent_directions.count(direction) > len(agent_directions) / 2:
                rationale_parts.append(f"多数Agent ({agent_directions.count(direction)}/{len(agent_directions)}) 支持{direction}")
            else:
                rationale_parts.append("基于加权投票结果做出决策")
        
        # 添加风险考虑
        risk_assessment = self._assess_risk(decision, market_context)
        if risk_assessment["risk_level"] == "High":
            rationale_parts.append(f"，尽管存在高风险因素: {', '.join(risk_assessment['risk_factors'])}")
        
        return "".join(rationale_parts)
    
    def _generate_mitigation_strategies(self, risk_factors: List[str]) -> List[str]:
        """生成风险缓解策略"""
        strategies = []
        
        if "High position size" in risk_factors:
            strategies.append("Consider reducing position size")
            strategies.append("Set tighter stop-loss")
        
        if "High market volatility" in risk_factors:
            strategies.append("Increase stop-loss distance")
            strategies.append("Reduce position size")
            strategies.append("Wait for volatility to decrease")
        
        if "Medium market volatility" in risk_factors:
            strategies.append("Use moderate position size")
            strategies.append("Set appropriate stop-loss")
        
        return strategies
    
    def save_explanations(self, output_dir: str):
        """保存解释到文件"""
        if not self.explanations:
            return
        
        explanations_path = os.path.join(output_dir, "risk_explanations.json")
        with open(explanations_path, 'w', encoding='utf-8') as f:
            json.dump(self.explanations, f, indent=2, ensure_ascii=False)
        
    def generate_summary_report(self) -> Dict:
        """生成摘要报告"""
        if not self.explanations:
            return {"message": "No explanations available"}
        
        # 统计决策分布
        decision_distribution = {"LONG": 0, "SHORT": 0, "HOLD": 0}
        risk_distribution = {"Low": 0, "Medium": 0, "High": 0}
        
        for explanation in self.explanations:
            decision = explanation.get("decision", {})
            direction = decision.get("direction", "HOLD")
            if direction in decision_distribution:
                decision_distribution[direction] += 1
            
            risk_assessment = explanation.get("risk_assessment", {})
            risk_level = risk_assessment.get("risk_level", "Low")
            if risk_level in risk_distribution:
                risk_distribution[risk_level] += 1
        
        return {
            "total_decisions": len(self.explanations),
            "decision_distribution": decision_distribution,
            "risk_distribution": risk_distribution,
            "average_confidence": self._calculate_average_confidence(),
            "most_common_risk_factors": self._identify_common_risk_factors()
        }
    
    def _calculate_average_confidence(self) -> float:
        """计算平均信心度"""
        total_confidence = 0.0
        count = 0
        
        for explanation in self.explanations:
            confidence_breakdown = explanation.get("confidence_breakdown", {})
            avg_conf = confidence_breakdown.get("average_confidence", 0.0)
            if avg_conf > 0:
                total_confidence += avg_conf
                count += 1
        
        return total_confidence / count if count > 0 else 0.0
    
    def _identify_common_risk_factors(self) -> List[str]:
        """识别常见风险因素"""
        factor_counts = {}
        
        for explanation in self.explanations:
            risk_assessment = explanation.get("risk_assessment", {})
            risk_factors = risk_assessment.get("risk_factors", [])
            for factor in risk_factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # 按频率排序
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        return [factor for factor, _ in sorted_factors[:3]]
