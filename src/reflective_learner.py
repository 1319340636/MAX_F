# ==========================================
# 文件名: src/reflective_learner.py
# 功能: 反思型自我演进模块 (Reflective Learner)
# 核心目标: 实现"回测中学习"，将盈亏结果转化为Agent的长期经验
# ==========================================

from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime
from src.memory import VectorMemoryManager

logger = logging.getLogger(__name__)

class ReflectiveLearner:
    """反思型自我演进模块
    
    实现基于回测反馈的自适应对齐，使Agent能够从交易结果中学习
    通过"裁判Agent"分析失败原因，生成交易准则并写入记忆系统
    """
    
    def __init__(self, vector_memory: Optional[VectorMemoryManager] = None):
        """初始化反思学习者
        
        Args:
            vector_memory: 向量记忆管理器实例
        """
        self.vector_memory = vector_memory or VectorMemoryManager()
        self.trade_reviews = []
        self.lessons_learned = []
        self.critic_llm = None  # 判官模型实例
        logger.info("✅ 反思型自我演进模块初始化完成")
    
    async def review_trade(self, trade_record: Dict[str, Any], agent_reasoning: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """审查交易并生成反思
        
        Args:
            trade_record: 交易记录
            agent_reasoning: Agent当时的推理过程
            market_data: 市场数据
            
        Returns:
            Dict: 反思结果
        """
        try:
            logger.info(f"📝 开始审查交易: PnL={trade_record.get('pnl_percent', 0):.2f}%")
            
            # 构造反思Prompt
            prompt = self._construct_reflection_prompt(trade_record, agent_reasoning, market_data)
            
            # 调用判官模型
            reflection = await self._invoke_critic_model(prompt)
            
            # 提取交易准则
            trading_rule = self._extract_trading_rule(reflection)
            
            # 写入向量记忆
            memory_id = await self._store_lesson_learned(trading_rule, trade_record, reflection)
            
            # 记录审查结果
            review_result = {
                "trade_id": trade_record.get('trade_id', f"trade_{datetime.now().timestamp()}"),
                "pnl_percent": trade_record.get('pnl_percent', 0),
                "timestamp": datetime.now().isoformat(),
                "reflection": reflection,
                "trading_rule": trading_rule,
                "memory_id": memory_id,
                "market_context": market_data.get('context', {})
            }
            
            self.trade_reviews.append(review_result)
            self.lessons_learned.append({
                "rule": trading_rule,
                "timestamp": datetime.now().isoformat(),
                "triggering_pnl": trade_record.get('pnl_percent', 0)
            })
            
            logger.info(f"✅ 交易审查完成，生成交易准则: {trading_rule[:100]}...")
            return review_result
            
        except Exception as e:
            logger.error(f"❌ 交易审查失败: {e}")
            return {
                "error": str(e),
                "trade_id": trade_record.get('trade_id', 'unknown')
            }
    
    def _construct_reflection_prompt(self, trade_record: Dict[str, Any], agent_reasoning: str, market_data: Dict[str, Any]) -> str:
        """构造反思提示词
        
        Args:
            trade_record: 交易记录
            agent_reasoning: Agent推理
            market_data: 市场数据
            
        Returns:
            str: 反思提示词
        """
        pnl_percent = trade_record.get('pnl_percent', 0)
        direction = trade_record.get('direction', 'UNKNOWN')
        entry_price = trade_record.get('entry_price', 0)
        exit_price = trade_record.get('exit_price', 0)
        
        market_move = "上涨" if (exit_price > entry_price and direction == "LONG") or (exit_price < entry_price and direction == "SHORT") else "下跌"
        
        volatility = market_data.get('volatility', 0)
        technical_pattern = market_data.get('technical_pattern', 'NONE')
        sentiment = market_data.get('sentiment', 'NEUTRAL')
        
        prompt = f"""
        # 交易审查任务
        
        ## 交易信息
        - 交易结果: {pnl_percent:.2f}%
        - 交易方向: {direction}
        - 入场价格: {entry_price}
        - 出场价格: {exit_price}
        - 市场后续走势: {market_move}
        - 市场波动率: {volatility:.2f}%
        - 技术形态: {technical_pattern}
        - 市场情绪: {sentiment}
        
        ## AI当时的决策理由
        {agent_reasoning}
        
        ## 任务要求
        1. 分析这次交易的逻辑是否存在缺陷
        2. 指出导致亏损的关键因素（如果是亏损交易）
        3. 生成一条具体、可操作的"交易准则"，用于指导未来类似行情
        4. 评估该准则的适用条件和局限性
        
        ## 输出格式
        请按照以下格式输出：
        
        ### 逻辑分析
        [分析AI决策逻辑的优点和缺陷]
        
        ### 亏损原因（如果适用）
        [详细分析导致亏损的因素]
        
        ### 交易准则
        [生成一条具体的交易准则]
        
        ### 适用条件
        [说明该准则的适用市场环境和条件]
        
        ### 局限性
        [说明该准则的局限性和注意事项]
        """
        
        return prompt
    
    async def _invoke_critic_model(self, prompt: str) -> str:
        """调用判官模型
        
        Args:
            prompt: 反思提示词
            
        Returns:
            str: 反思结果
        """
        # 这里应该调用实际的LLM模型
        # 由于没有实际的模型实例，返回模拟结果
        # 实际实现时需要替换为真实的LLM调用
        
        # 模拟判官模型的响应
        await asyncio.sleep(1)  # 模拟网络延迟
        
        # 生成模拟反思
        mock_reflection = """
        ### 逻辑分析
        AI的决策逻辑基于技术形态分析，识别到了突破信号，这是合理的。然而，忽略了市场情绪处于极度乐观状态这一重要因素，导致在高点入场。
        
        ### 亏损原因
        1. 入场时机不佳：在市场情绪极度乐观时入场，价格已接近短期顶部
        2. 止损设置不合理：止损距离过窄，容易被正常波动触发
        3. 忽略了波动率因素：高波动率环境下，突破信号的可靠性降低
        
        ### 交易准则
        在市场情绪极度乐观（sentiment > 0.8）且波动率高于20%时，对于突破形态的多头信号应保持谨慎，建议观望或只进行小仓位试探。
        
        ### 适用条件
        - 市场情绪指标 > 0.8
        - 波动率 > 20%
        - 技术形态显示突破信号
        - 短期价格已大幅上涨
        
        ### 局限性
        - 不适用于低波动率环境
        - 不适用于市场情绪中性或悲观的情况
        - 在重大利好消息驱动的突破中可能过于保守
        """
        
        return mock_reflection
    
    def _extract_trading_rule(self, reflection: str) -> str:
        """从反思中提取交易准则
        
        Args:
            reflection: 反思结果
            
        Returns:
            str: 交易准则
        """
        # 提取交易准则部分
        lines = reflection.split('\n')
        in_rule_section = False
        rule_lines = []
        
        for line in lines:
            if '### 交易准则' in line:
                in_rule_section = True
                continue
            elif line.startswith('### ') and in_rule_section:
                break
            elif in_rule_section and line.strip():
                rule_lines.append(line.strip())
        
        trading_rule = ' '.join(rule_lines)
        return trading_rule if trading_rule else "无明确交易准则"
    
    async def _store_lesson_learned(self, trading_rule: str, trade_record: Dict[str, Any], reflection: str) -> str:
        """存储学到的教训到向量记忆
        
        Args:
            trading_rule: 交易准则
            trade_record: 交易记录
            reflection: 反思结果
            
        Returns:
            str: 记忆ID
        """
        # 构造记忆内容
        memory_content = f"""
        # 交易教训
        
        ## 交易准则
        {trading_rule}
        
        ## 触发场景
        - 交易结果: {trade_record.get('pnl_percent', 0):.2f}%
        - 交易方向: {trade_record.get('direction', 'UNKNOWN')}
        - 市场环境: {trade_record.get('market_environment', 'UNKNOWN')}
        
        ## 详细分析
        {reflection}
        """
        
        # 构造元数据
        metadata = {
            "type": "lesson_learned",
            "timestamp": datetime.now().isoformat(),
            "pnl_percent": trade_record.get('pnl_percent', 0),
            "direction": trade_record.get('direction', 'UNKNOWN'),
            "market_environment": trade_record.get('market_environment', 'UNKNOWN'),
            "importance": "HIGH" if abs(trade_record.get('pnl_percent', 0)) > 2 else "MEDIUM"
        }
        
        # 写入向量记忆
        try:
            # 将memory_type添加到metadata中
            metadata["memory_type"] = "lesson"
            # 调用add_memory方法，只传递content和metadata参数
            self.vector_memory.add_memory(
                content=memory_content,
                metadata=metadata
            )
            logger.info("📚 交易教训已存储到向量记忆")
            return f"memory_{datetime.now().timestamp()}"
        except Exception as e:
            logger.error(f"❌ 存储交易教训失败: {e}")
            return f"error_{datetime.now().timestamp()}"
    
    async def batch_review_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量审查交易
        
        Args:
            trades: 交易记录列表
            
        Returns:
            List: 审查结果列表
        """
        tasks = []
        for trade in trades:
            # 过滤需要审查的交易（主要是亏损交易）
            pnl = trade.get('pnl_percent', 0)
            if pnl < -1.0:  # 只审查亏损超过1%的交易
                task = self.review_trade(
                    trade_record=trade,
                    agent_reasoning=trade.get('agent_reasoning', '无推理记录'),
                    market_data=trade.get('market_data', {})
                )
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks)
            logger.info(f"✅ 批量审查完成，共审查 {len(results)} 笔交易")
            return results
        else:
            logger.info("ℹ️  没有需要审查的交易")
            return []
    
    def get_lessons_summary(self) -> Dict[str, Any]:
        """获取学到的教训摘要
        
        Returns:
            Dict: 教训摘要
        """
        summary = {
            "total_trade_reviews": len(self.trade_reviews),
            "total_lessons_learned": len(self.lessons_learned),
            "lessons_by_importance": {
                "HIGH": sum(1 for lesson in self.lessons_learned if abs(lesson.get('triggering_pnl', 0)) > 2),
                "MEDIUM": sum(1 for lesson in self.lessons_learned if 1 <= abs(lesson.get('triggering_pnl', 0)) <= 2),
                "LOW": sum(1 for lesson in self.lessons_learned if abs(lesson.get('triggering_pnl', 0)) < 1)
            },
            "recent_lessons": self.lessons_learned[-5:] if self.lessons_learned else []
        }
        
        return summary
    
    def reset(self):
        """重置反思学习者"""
        self.trade_reviews = []
        self.lessons_learned = []
        logger.info("🔄 反思型自我演进模块已重置")

# 全局反思学习者实例
reflective_learner = ReflectiveLearner()
