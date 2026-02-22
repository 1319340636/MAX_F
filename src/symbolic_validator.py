# ==========================================
# 文件名: src/symbolic_validator.py
# 功能: 逻辑一致性校验器 (Symbolic Validator)
# 核心目标: 消除神经模型的"幻觉"，确保AI的直觉结论符合物理数学事实
# ==========================================

from typing import Optional, Dict, Any
from src.models import Prediction, MarketContext
import logging

logger = logging.getLogger(__name__)

class SymbolicValidator:
    """逻辑一致性校验器"""
    
    def __init__(self):
        self.validation_history = []
        self.failed_validations = 0
        self.total_validations = 0
    
    def validate(self, prediction: Prediction, context: MarketContext, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证预测的逻辑一致性
        
        Args:
            prediction: Agent的预测结果
            context: 市场上下文
            market_data: 额外的市场数据，包含技术指标等
            
        Returns:
            Dict: 包含验证结果和详细信息
        """
        self.total_validations += 1
        
        validation_result = {
            "valid": True,
            "reasons": [],
            "corrections": [],
            "confidence_adjustment": 0.0
        }
        
        reasoning = prediction.reasoning.lower() if prediction.reasoning else ""
        
        # 逻辑1：校验"突破"真实性
        if "突破" in reasoning or "breakout" in reasoning:
            validation_result = self._validate_breakout(prediction, context, market_data, validation_result)
        
        # 逻辑2：校验"超卖/超买"一致性
        if "超卖" in reasoning or "oversold" in reasoning:
            validation_result = self._validate_oversold(prediction, context, market_data, validation_result)
        
        if "超买" in reasoning or "overbought" in reasoning:
            validation_result = self._validate_overbought(prediction, context, market_data, validation_result)
        
        # 逻辑3：校验"趋势"一致性
        if "趋势" in reasoning or "trend" in reasoning:
            validation_result = self._validate_trend(prediction, context, market_data, validation_result)
        
        # 逻辑4：校验"支撑/阻力"一致性
        if "支撑" in reasoning or "support" in reasoning:
            validation_result = self._validate_support(prediction, context, market_data, validation_result)
        
        if "阻力" in reasoning or "resistance" in reasoning:
            validation_result = self._validate_resistance(prediction, context, market_data, validation_result)
        
        # 逻辑5：GVZ 阈值符号熔断机制
        validation_result = self._validate_gvz_threshold(prediction, context, market_data, validation_result)
        
        # 记录验证结果
        validation_record = {
            "agent_id": prediction.agent_id,
            "direction": prediction.direction,
            "confidence": prediction.confidence,
            "valid": validation_result["valid"],
            "reasons": validation_result["reasons"],
            "timestamp": context.timestamp
        }
        
        self.validation_history.append(validation_record)
        
        if not validation_result["valid"]:
            self.failed_validations += 1
            logger.warning(f"❌ 验证失败: {prediction.agent_id} 的预测不符合数学事实")
            logger.warning(f"   理由: {prediction.reasoning}")
            logger.warning(f"   纠正: {validation_result['corrections']}")
        else:
            logger.info(f"✅ 验证通过: {prediction.agent_id} 的预测符合数学事实")
        
        return validation_result
    
    def _contains_negation(self, text: str) -> bool:
        """检测文本中是否包含否定词"""
        negation_words = ['未', '没有', '不', '无', 'not', 'no', 'fail', '没有形成', '未形成', '未能', '没有突破', '未突破']
        for word in negation_words:
            if word in text:
                return True
        return False
    
    def _validate_breakout(self, prediction: Prediction, context: MarketContext, market_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """验证突破的真实性"""
        # 检查当前价格是否真的高于过去N天的最高价
        high_n_days = market_data.get('high_n_days', 0)
        current_price = context.price or context.close if hasattr(context, 'close') else 0
        
        # 检测是否包含否定词
        if self._contains_negation(prediction.reasoning):
            # 如果包含否定词，说明模型在否定突破，不需要验证
            result["reasons"].append("突破验证通过：模型在否定突破，符合当前市场状态")
        else:
            # 只在模型肯定突破时进行验证
            if current_price <= high_n_days:
                result["valid"] = False
                result["reasons"].append("突破验证失败：当前价格未超过过去N天最高价")
                result["corrections"].append("你说的突破并不符合数学事实，请重新观察")
                result["confidence_adjustment"] -= 0.3
            else:
                result["reasons"].append("突破验证通过：当前价格确实超过过去N天最高价")
        
        return result
    
    def _validate_oversold(self, prediction: Prediction, context: MarketContext, market_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """验证超卖的一致性"""
        rsi = market_data.get('rsi', 50)
        
        # 检测是否包含否定词
        if self._contains_negation(prediction.reasoning):
            # 如果包含否定词，说明模型在否定超卖，不需要验证
            result["reasons"].append("超卖验证通过：模型在否定超卖，符合当前市场状态")
        else:
            # 只在模型肯定超卖时进行验证
            if rsi > 35:
                result["valid"] = False
                result["reasons"].append("超卖验证失败：RSI未低于超卖阈值")
                result["corrections"].append("当前市场并未处于超卖状态，请重新评估")
                result["confidence_adjustment"] -= 0.2
            else:
                result["reasons"].append("超卖验证通过：RSI确实处于超卖区间")
        
        return result
    
    def _validate_overbought(self, prediction: Prediction, context: MarketContext, market_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """验证超买的一致性"""
        rsi = market_data.get('rsi', 50)
        
        # 检测是否包含否定词
        if self._contains_negation(prediction.reasoning):
            # 如果包含否定词，说明模型在否定超买，不需要验证
            result["reasons"].append("超买验证通过：模型在否定超买，符合当前市场状态")
        else:
            # 只在模型肯定超买时进行验证
            if rsi < 65:
                result["valid"] = False
                result["reasons"].append("超买验证失败：RSI未高于超买阈值")
                result["corrections"].append("当前市场并未处于超买状态，请重新评估")
                result["confidence_adjustment"] -= 0.2
            else:
                result["reasons"].append("超买验证通过：RSI确实处于超买区间")
        
        return result
    
    def _validate_trend(self, prediction: Prediction, context: MarketContext, market_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """验证趋势的一致性"""
        ma20 = market_data.get('ma20', 0)
        ma60 = market_data.get('ma60', 0)
        current_price = context.price or context.close if hasattr(context, 'close') else 0
        
        # 检测是否包含否定词
        if self._contains_negation(prediction.reasoning):
            # 如果包含否定词，说明模型在否定趋势，不需要验证
            result["reasons"].append("趋势验证通过：模型在否定趋势，符合当前市场状态")
        else:
            # 只在模型肯定趋势时进行验证
            if ma20 > 0 and ma60 > 0:
                if "多头" in prediction.reasoning or "bull" in prediction.reasoning.lower():
                    if not (current_price > ma20 > ma60):
                        result["valid"] = False
                        result["reasons"].append("多头趋势验证失败：价格未在均线上方且均线未形成多头排列")
                        result["corrections"].append("当前市场并未形成明显的多头趋势，请重新分析")
                        result["confidence_adjustment"] -= 0.25
                elif "空头" in prediction.reasoning or "bear" in prediction.reasoning.lower():
                    if not (current_price < ma20 < ma60):
                        result["valid"] = False
                        result["reasons"].append("空头趋势验证失败：价格未在均线下方且均线未形成空头排列")
                        result["corrections"].append("当前市场并未形成明显的空头趋势，请重新分析")
                        result["confidence_adjustment"] -= 0.25
        
        return result
    
    def _validate_support(self, prediction: Prediction, context: MarketContext, market_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """验证支撑位的一致性"""
        support_level = market_data.get('support_level', 0)
        current_price = context.price or context.close if hasattr(context, 'close') else 0
        
        # 检测是否包含否定词
        if self._contains_negation(prediction.reasoning):
            # 如果包含否定词，说明模型在否定支撑位相关表述，不需要验证
            result["reasons"].append("支撑位验证通过：模型在否定支撑位相关表述，符合当前市场状态")
        else:
            # 只在模型肯定支撑位时进行验证
            if support_level > 0:
                if current_price < support_level - (support_level * 0.01):  # 允许1%误差
                    result["valid"] = False
                    result["reasons"].append("支撑位验证失败：当前价格已跌破支撑位")
                    result["corrections"].append("价格已跌破你提到的支撑位，请重新评估市场状态")
                    result["confidence_adjustment"] -= 0.2
        
        return result
    
    def _validate_resistance(self, prediction: Prediction, context: MarketContext, market_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """验证阻力位的一致性"""
        resistance_level = market_data.get('resistance_level', 0)
        current_price = context.price or context.close if hasattr(context, 'close') else 0
        
        # 检测是否包含否定词
        if self._contains_negation(prediction.reasoning):
            # 如果包含否定词，说明模型在否定阻力位相关表述，不需要验证
            result["reasons"].append("阻力位验证通过：模型在否定阻力位相关表述，符合当前市场状态")
        else:
            # 只在模型肯定阻力位时进行验证
            if resistance_level > 0:
                if current_price > resistance_level + (resistance_level * 0.01):  # 允许1%误差
                    result["valid"] = False
                    result["reasons"].append("阻力位验证失败：当前价格已突破阻力位")
                    result["corrections"].append("价格已突破你提到的阻力位，请重新评估市场状态")
                    result["confidence_adjustment"] -= 0.2
        
        return result
    
    def _validate_gvz_threshold(self, prediction: Prediction, context: MarketContext, market_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """GVZ 阈值符号熔断机制
        
        如果 GVZ > 30（极度恐慌），无论 Technical_Analyst 信心多高，
        Risk_Manager 必须强制将 single_position_pct 降至 5% 以下。
        """
        # 获取 GVZ 数据
        gvz = market_data.get('GVZ', 0)
        gvz_zscore = market_data.get('GVZ_ZScore', 0)
        
        # 检查 Risk_Manager 的仓位建议
        if prediction.agent_id == "Risk":
            position_size = getattr(prediction, 'stake', 0) or getattr(prediction, 'position_size', 0)
            
            # 极度恐慌状态：GVZ > 30 或 GVZ Z-Score > 3
            if (gvz > 30) or (gvz_zscore > 3):
                if position_size > 50:  # 50 对应 5% 仓位
                    result["valid"] = False
                    result["reasons"].append("GVZ 阈值熔断：当前处于极度恐慌状态，必须降低仓位")
                    result["corrections"].append("当前 GVZ 显示极度恐慌，必须将仓位降至 5% 以下")
                    result["confidence_adjustment"] -= 0.4
                    
                    # 强制调整仓位建议
                    if hasattr(prediction, 'stake'):
                        prediction.stake = 50  # 5% 仓位
                    if hasattr(prediction, 'position_size'):
                        prediction.position_size = 0.05  # 5% 仓位
        
        # 检查其他代理的过度乐观
        elif prediction.direction == "LONG" and prediction.confidence > 0.8:
            if (gvz > 25) or (gvz_zscore > 2.5):
                result["valid"] = False
                result["reasons"].append("GVZ 警告：在高恐慌状态下过度乐观")
                result["corrections"].append("当前市场恐慌程度较高，建议降低多头信心")
                result["confidence_adjustment"] -= 0.3
        
        return result
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        if self.total_validations == 0:
            return {
                "total_validations": 0,
                "failed_validations": 0,
                "success_rate": 0.0
            }
        
        return {
            "total_validations": self.total_validations,
            "failed_validations": self.failed_validations,
            "success_rate": (self.total_validations - self.failed_validations) / self.total_validations * 100
        }
    
    def reset_stats(self):
        """重置验证统计信息"""
        self.validation_history = []
        self.failed_validations = 0
        self.total_validations = 0
