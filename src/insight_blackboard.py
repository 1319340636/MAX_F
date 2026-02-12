# ==========================================
# 文件名: src/insight_blackboard.py
# 功能: 跨代理协作公告板 (Collaborative Blackboard)
# 核心目标: 实现Agent间的"认知对齐"，从孤立决策转向群体博弈
# ==========================================

from typing import Dict, Any, Optional
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

class InsightBlackboard:
    """跨代理协作公告板
    
    实现Agent间的信息共享，建立全局认知对齐机制
    每个Agent在决策前先读取，决策后写入自己的高维洞察
    """
    
    # 单例模式
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InsightBlackboard, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化公告板"""
        self.insights = {
            # 宏观层面洞察
            "macro_bias": "NEUTRAL",  # 宏观偏见: BULLISH, BEARISH, NEUTRAL
            "macro_factors": [],      # 宏观影响因素
            "interest_rate_outlook": "STABLE",  # 利率展望
            "inflation_expectation": "MODERATE",  # 通胀预期
            
            # 技术层面洞察
            "major_resistance": 0.0,   # 主要阻力位
            "major_support": 0.0,      # 主要支撑位
            "technical_trend": "SIDEways",  # 技术趋势
            "key_technical_levels": [],  # 关键技术位
            "chart_pattern": "NONE",    # 图表形态
            
            # 市场情绪洞察
            "sentiment_score": 0.5,     # 情绪评分: 0-1 (0=极度悲观, 1=极度乐观)
            "sentiment_extreme": False,  # 是否处于情绪极值
            "market_fear_greed": 50,     # 恐慌贪婪指数: 0-100
            "retail_sentiment": "NEUTRAL",  # 散户情绪
            
            # 风险管理洞察
            "volatility_regime": "NORMAL",  # 波动率 regime
            "risk_level": "LOW",        # 风险等级
            "liquidity_condition": "ADEQUATE",  # 流动性状况
            "correlation_pattern": "NORMAL",  # 相关性模式
            
            # 时间戳和更新信息
            "last_updated": datetime.now().isoformat(),
            "last_updater": "SYSTEM",
            "insight_history": []
        }
        
        self.role_insights = {
            "Macro": {},  # 宏观分析师洞察
            "Tech": {},   # 技术分析师洞察
            "Risk": {},   # 风险管理者洞察
            "Sent": {},   # 情绪分析师洞察
            "Fund": {}    # 基本面分析师洞察
        }
        
        self._lock = threading.RLock()
        logger.info("✅ 洞察公告板初始化完成")
    
    def update_insight(self, role: str, data: Dict[str, Any]) -> bool:
        """更新洞察信息
        
        Args:
            role: 更新者的角色 (Macro, Tech, Risk, Sent, Fund)
            data: 要更新的洞察数据
            
        Returns:
            bool: 更新是否成功
        """
        with self._lock:
            try:
                # 记录历史洞察
                self._record_history(role, data)
                
                # 更新对应角色的洞察
                self.role_insights[role].update(data)
                
                # 根据角色更新全局洞察
                if role == "Macro":
                    self._update_macro_insights(data)
                elif role == "Tech":
                    self._update_tech_insights(data)
                elif role == "Risk":
                    self._update_risk_insights(data)
                elif role == "Sent":
                    self._update_sentiment_insights(data)
                elif role == "Fund":
                    self._update_fundamental_insights(data)
                
                # 更新时间戳
                self.insights["last_updated"] = datetime.now().isoformat()
                self.insights["last_updater"] = role
                
                logger.info(f"📢 {role} 更新了洞察公告板: {data}")
                return True
                
            except Exception as e:
                logger.error(f"❌ 更新洞察失败: {e}")
                return False
    
    def get_insights(self, role: Optional[str] = None) -> Dict[str, Any]:
        """获取洞察信息
        
        Args:
            role: 可选，指定角色，获取该角色的专属洞察
            
        Returns:
            Dict: 洞察信息
        """
        with self._lock:
            if role and role in self.role_insights:
                # 返回全局洞察 + 角色专属洞察
                combined_insights = self.insights.copy()
                combined_insights["role_specific"] = self.role_insights[role].copy()
                return combined_insights
            else:
                # 返回全局洞察
                return self.insights.copy()
    
    def get_cross_role_insights(self) -> Dict[str, Any]:
        """获取跨角色的综合洞察
        
        Returns:
            Dict: 包含所有角色洞察的综合信息
        """
        with self._lock:
            return {
                "global_insights": self.insights.copy(),
                "role_specific_insights": self.role_insights.copy()
            }
    
    def reset(self):
        """重置公告板"""
        with self._lock:
            self._initialize()
            logger.info("🔄 洞察公告板已重置")
    
    def _record_history(self, role: str, data: Dict[str, Any]):
        """记录洞察历史"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "insights": data.copy(),
            "context": {
                "macro_bias": self.insights["macro_bias"],
                "technical_trend": self.insights["technical_trend"],
                "sentiment_score": self.insights["sentiment_score"]
            }
        }
        
        self.insights["insight_history"].append(history_entry)
        
        # 限制历史记录长度
        if len(self.insights["insight_history"]) > 100:
            self.insights["insight_history"] = self.insights["insight_history"][-100:]
    
    def _update_macro_insights(self, data: Dict[str, Any]):
        """更新宏观洞察"""
        if "bias" in data:
            self.insights["macro_bias"] = data["bias"]
        if "factors" in data:
            self.insights["macro_factors"] = data["factors"]
        if "interest_rate_outlook" in data:
            self.insights["interest_rate_outlook"] = data["interest_rate_outlook"]
        if "inflation_expectation" in data:
            self.insights["inflation_expectation"] = data["inflation_expectation"]
    
    def _update_tech_insights(self, data: Dict[str, Any]):
        """更新技术洞察"""
        if "resistance" in data:
            self.insights["major_resistance"] = data["resistance"]
        if "support" in data:
            self.insights["major_support"] = data["support"]
        if "trend" in data:
            self.insights["technical_trend"] = data["trend"]
        if "key_levels" in data:
            self.insights["key_technical_levels"] = data["key_levels"]
        if "pattern" in data:
            self.insights["chart_pattern"] = data["pattern"]
    
    def _update_risk_insights(self, data: Dict[str, Any]):
        """更新风险洞察"""
        if "volatility_regime" in data:
            self.insights["volatility_regime"] = data["volatility_regime"]
        if "risk_level" in data:
            self.insights["risk_level"] = data["risk_level"]
        if "liquidity" in data:
            self.insights["liquidity_condition"] = data["liquidity"]
        if "correlation" in data:
            self.insights["correlation_pattern"] = data["correlation"]
    
    def _update_sentiment_insights(self, data: Dict[str, Any]):
        """更新情绪洞察"""
        if "sentiment_score" in data:
            self.insights["sentiment_score"] = data["sentiment_score"]
        if "sentiment_extreme" in data:
            self.insights["sentiment_extreme"] = data["sentiment_extreme"]
        if "fear_greed" in data:
            self.insights["market_fear_greed"] = data["fear_greed"]
        if "retail_sentiment" in data:
            self.insights["retail_sentiment"] = data["retail_sentiment"]
    
    def _update_fundamental_insights(self, data: Dict[str, Any]):
        """更新基本面洞察"""
        # 基本面洞察可以影响多个领域
        if "valuation" in data:
            # 估值信息可以影响宏观偏见
            if data["valuation"] == "OVERVALUED":
                self.insights["macro_bias"] = "BEARISH"
            elif data["valuation"] == "UNDERVALUED":
                self.insights["macro_bias"] = "BULLISH"
        
        if "earnings_outlook" in data:
            # 盈利展望可以影响市场情绪
            if data["earnings_outlook"] == "POSITIVE":
                self.insights["sentiment_score"] = min(1.0, self.insights["sentiment_score"] + 0.2)
            elif data["earnings_outlook"] == "NEGATIVE":
                self.insights["sentiment_score"] = max(0.0, self.insights["sentiment_score"] - 0.2)
    
    def get_market_context(self) -> Dict[str, Any]:
        """获取市场上下文摘要
        
        Returns:
            Dict: 市场上下文摘要，适合作为Agent决策的背景信息
        """
        with self._lock:
            return {
                "macro_bias": self.insights["macro_bias"],
                "technical_trend": self.insights["technical_trend"],
                "sentiment_score": self.insights["sentiment_score"],
                "risk_level": self.insights["risk_level"],
                "major_resistance": self.insights["major_resistance"],
                "major_support": self.insights["major_support"],
                "volatility_regime": self.insights["volatility_regime"],
                "last_updated": self.insights["last_updated"]
            }
    
    def get_insight_summary(self) -> str:
        """获取洞察摘要
        
        Returns:
            str: 洞察摘要文本，适合在Agent提示中使用
        """
        context = self.get_market_context()
        
        summary_parts = []
        
        if context["macro_bias"] != "NEUTRAL":
            summary_parts.append(f"宏观环境: {context['macro_bias']}")
        
        if context["technical_trend"] != "SIDEways":
            summary_parts.append(f"技术趋势: {context['technical_trend']}")
        
        sentiment = context["sentiment_score"]
        if sentiment > 0.7:
            summary_parts.append("市场情绪: 极度乐观")
        elif sentiment > 0.5:
            summary_parts.append("市场情绪: 乐观")
        elif sentiment < 0.3:
            summary_parts.append("市场情绪: 极度悲观")
        elif sentiment < 0.5:
            summary_parts.append("市场情绪: 悲观")
        
        if context["risk_level"] != "LOW":
            summary_parts.append(f"风险等级: {context['risk_level']}")
        
        if context["volatility_regime"] != "NORMAL":
            summary_parts.append(f"波动率状态: {context['volatility_regime']}")
        
        if context["major_resistance"] > 0:
            summary_parts.append(f"主要阻力位: {context['major_resistance']}")
        
        if context["major_support"] > 0:
            summary_parts.append(f"主要支撑位: {context['major_support']}")
        
        if summary_parts:
            return "；".join(summary_parts)
        else:
            return "市场处于中性状态，无明显主导因素"

# 全局公告板实例
blackboard = InsightBlackboard()
