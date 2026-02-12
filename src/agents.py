# ==========================================
# 文件名: src/agents.py
# ==========================================

import json
import re
import io
import os
import base64
import logging
import asyncio 
from typing import Union, Optional

# LangChain
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# 本地模块
from src.models import Prediction, MarketContext
# ✅ 关键修改：从同一个文件 src.memory 导入两个管理器
from src.memory import TieredMemoryManager, VectorMemoryManager
from src.prompts import (
    get_role_system_prompt, 
    DECISION_TASK_TEMPLATE, 
    REFLECTION_TASK_TEMPLATE
)
# ✅ 新增：导入新模块
from src.symbolic_validator import SymbolicValidator
from src.insight_blackboard import blackboard
from src.reflective_learner import reflective_learner

logger = logging.getLogger(__name__)

# 定义原则
COMMON_PRINCIPLES = [
    "风险控制是盈利的基础，适度冒险是收益的来源。",
    "严格遵守系统设定的 Hard Limits，禁止手动突破。"
]
TREND_PRINCIPLES = [
    "趋势初期适度参与，趋势中期稳健持有，趋势末期逐步退出。",
    "MA60向上+价格在MA20上方：默认做多环境。"
]

class HybridAgent:
    def __init__(self, agent_id: str, role: str, symbol: str = "GC=F"):
        self.agent_id = agent_id
        self.role = role
        self.symbol = symbol
        
        # 1. 初始化记忆系统 (Tiered)
        self.memory_system = TieredMemoryManager(agent_id, role, symbol)
        
        # 2. 初始化向量记忆 (Vector) - 路径已修正
        self.vector_mem = VectorMemoryManager(storage_path=f"memory_store/{agent_id}")
        self.vector_mem.load() 
        
        # 3. 加载原则
        self.long_term_principles = list(COMMON_PRINCIPLES)
        if role == "Technical_Analyst":
            self.long_term_principles += TREND_PRINCIPLES
        
        # 4. 初始化符号验证器
        self.validator = SymbolicValidator()

        # 4. 初始化 LLM
        if role == "Technical_Analyst":
            self.model_type = "local_vision"
            self.llm = ChatOllama(
                model="qwen3-vl:8b", 
                temperature=0.3, 
                keep_alive="24h"
            )
        elif role == "Sentiment_Watcher":
            self.model_type = "local_text"
            self.llm = ChatOllama(
                model="qwen3:8b", 
                temperature=0.5, 
                num_ctx=16384, 
                keep_alive="24h"
            )
        else:
            self.model_type = "cloud_reasoning"
            self.llm = ChatOpenAI(
                model="Pro/deepseek-ai/DeepSeek-V3.2", 
                api_key="sk-xbobpvwcbzocotlefnqkgjusmxpejoeguqynqymzeauncwyc", 
                base_url="https://api.siliconflow.cn/v1", 
                temperature=0.6
            )

    async def decide(self, context: MarketContext, chart_path: Union[str, io.BytesIO, None] = None, market_data: Optional[dict] = None) -> Prediction:
        """异步决策函数"""
        # A. 读取洞察公告板
        market_context_summary = blackboard.get_insight_summary()
        board_insights = f"\n【📊 市场共识】:{market_context_summary}"
        
        # B. 记忆检索
        basic_lessons = self.memory_system.retrieve_working_memory()
        
        # 向量检索
        query = f"{context.market_env} {context.news_summary} {market_context_summary}"
        similar_mems = self.vector_mem.find_similar_memories(query, top_k=3)
        vector_lessons = ""
        if similar_mems:
            vector_lessons = "\n【🔮 历史相似场景】:\n" + "\n".join(
                [f"- {m['content']} (相似度:{m['similarity']:.2f})" for m in similar_mems]
            )
        combined_lessons = f"{basic_lessons}\n{vector_lessons}\n{board_insights}"

        # C. 构造 Prompt
        role_sys_prompt = get_role_system_prompt(self.role, context, lessons_text=combined_lessons)
        final_prompt = DECISION_TASK_TEMPLATE.format(
            role_prompt=role_sys_prompt,
            principles_text="\n".join([f"- {p}" for p in self.long_term_principles]),
            lessons_text=combined_lessons
        )

        message = final_prompt
        if self.model_type == "local_vision" and chart_path:
            img_b64 = self._encode_image(chart_path)
            if img_b64:
                message = [HumanMessage(content=[
                    {"type": "text", "text": final_prompt + "\n【技术面必读】请参考K线图："},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ])]

        try:
            res = await self.llm.ainvoke(message)
            content = self._clean_think_tag(res.content)
            
            try:
                data = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
            except:
                data = {"direction": "HOLD", "confidence": 0.0, "reasoning": "Parse Error"}

            # 创建预测
            prediction = Prediction(
                agent_id=self.agent_id,
                direction=str(data.get("direction", "HOLD")).upper(),
                confidence=self._safe_float(data.get("confidence"), 0.5),
                stake=self._safe_float(data.get("stake"), 0.0),
                reasoning=data.get("reasoning", "")[:200],
                stop_loss=self._safe_float(data.get("stop_loss"), 0.0),
                take_profit=self._safe_float(data.get("take_profit"), 0.0),
                position_ratio=self._safe_float(data.get("position_ratio"), 0.0)
            )
            
            # D. 符号验证
            if market_data:
                validation_result = self.validator.validate(prediction, context, market_data)
                if not validation_result["valid"]:
                    logger.warning(f"⚠️  {self.agent_id} 预测验证失败，需要重新评估")
                    # 调整信心度
                    prediction.confidence = max(0.1, prediction.confidence + validation_result["confidence_adjustment"])
                    # 添加验证反馈到推理过程
                    prediction.reasoning += " [验证反馈: " + "; ".join(validation_result["corrections"]) + "]"
            
            # E. 更新洞察公告板
            self._update_blackboard(prediction, context, market_data)
            
            return prediction
        except Exception as e:
            logger.error(f"Agent {self.agent_id} decide error: {e}")
            error_prediction = Prediction(self.agent_id, "HOLD", 0.0, 0.0, f"Error: {str(e)[:50]}")
            # 更新公告板记录错误
            self._update_blackboard(error_prediction, context, None)
            return error_prediction

    async def reflect(self, context, direction, outcome, actual_chg, confidence=None, stake=None, market_data=None):
        """异步反思函数"""
        if actual_chg > 0.01:
            self.memory_system.process_feedback("", True, actual_chg, context.volatility)
            return

        stop_loss_pct = 0.02
        if direction != outcome or abs(actual_chg) > stop_loss_pct or (confidence and confidence > 0.8):
            prompt = REFLECTION_TASK_TEMPLATE.format(
                role=self.role, 
                symbol=context.symbol,
                market_env=f"{context.market_env} (Vol:{context.volatility:.2%})",
                my_action=direction, actual_outcome=outcome, actual_return=actual_chg
            )
            try:
                res = await self.llm.ainvoke(prompt)
                lesson = self._clean_think_tag(res.content)
                
                # 存入双重记忆
                self.memory_system.process_feedback(f"[{context.market_env}] {lesson}", False, actual_chg, context.volatility)
                self.vector_mem.add_memory(f"环境:{context.market_env}, 决策:{direction}, 结果:{outcome}, 教训:{lesson}")
                self.vector_mem.save()
                
                # 调用反思学习者进行深度审查
                if market_data:
                    trade_record = {
                        'trade_id': f"{self.agent_id}_{context.timestamp}",
                        'direction': direction,
                        'outcome': outcome,
                        'pnl_percent': actual_chg * 100,
                        'market_environment': context.market_env,
                        'volatility': context.volatility
                    }
                    
                    # 异步审查交易
                    await reflective_learner.review_trade(
                        trade_record=trade_record,
                        agent_reasoning=lesson,
                        market_data=market_data
                    )
                    
            except Exception as e:
                logger.error(f"Agent {self.agent_id} reflect error: {e}")
        else:
            self.memory_system.process_feedback("", False, actual_chg, context.volatility)

    def _encode_image(self, chart_source):
        try:
            if isinstance(chart_source, io.BytesIO): 
                return base64.b64encode(chart_source.getvalue()).decode("utf-8")
            elif isinstance(chart_source, str):
                # 检查文件是否存在
                if os.path.exists(chart_source):
                    # 检查文件大小，确保不是空文件
                    if os.path.getsize(chart_source) > 0:
                        with open(chart_source, "rb") as f:
                            return base64.b64encode(f.read()).decode("utf-8")
                    else:
                        logger.warning(f"⚠️  图片文件为空: {chart_source}")
                else:
                    logger.warning(f"⚠️  图片文件不存在: {chart_source}")
        except Exception as e:
            logger.error(f"❌ 图片编码失败: {e}")
        return None

    def _clean_think_tag(self, text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.replace("```json", "").replace("```", "").strip()

    def _safe_float(self, val, default=0.0):
        try: return float(val) if val is not None else default
        except: return default
    
    def _update_blackboard(self, prediction: Prediction, context: MarketContext, market_data: Optional[dict]):
        """更新洞察公告板
        
        Args:
            prediction: Agent的预测
            context: 市场上下文
            market_data: 市场数据
        """
        try:
            # 根据Agent角色更新不同类型的洞察
            if self.agent_id == "Tech":
                # 技术分析师更新技术层面洞察
                tech_insights = {
                    "technical_trend": prediction.direction,
                    "confidence": prediction.confidence,
                    "reasoning": prediction.reasoning
                }
                if market_data:
                    if "resistance_level" in market_data:
                        tech_insights["resistance"] = market_data["resistance_level"]
                    if "support_level" in market_data:
                        tech_insights["support"] = market_data["support_level"]
                    if "chart_pattern" in market_data:
                        tech_insights["pattern"] = market_data["chart_pattern"]
                blackboard.update_insight("Tech", tech_insights)
            
            elif self.agent_id == "Macro":
                # 宏观分析师更新宏观层面洞察
                macro_insights = {
                    "bias": prediction.direction,
                    "confidence": prediction.confidence,
                    "factors": [prediction.reasoning[:100]]
                }
                blackboard.update_insight("Macro", macro_insights)
            
            elif self.agent_id == "Risk":
                # 风险管理者更新风险管理洞察
                risk_insights = {
                    "risk_level": "HIGH" if prediction.confidence > 0.8 else "MEDIUM",
                    "volatility_regime": context.volatility,
                    "position_size": prediction.stake
                }
                blackboard.update_insight("Risk", risk_insights)
            
            elif self.agent_id == "Sent":
                # 情绪分析师更新市场情绪洞察
                sentiment_score = 0.5
                if prediction.direction == "LONG":
                    sentiment_score = 0.5 + (prediction.confidence * 0.4)
                elif prediction.direction == "SHORT":
                    sentiment_score = 0.5 - (prediction.confidence * 0.4)
                
                sent_insights = {
                    "sentiment_score": sentiment_score,
                    "sentiment_extreme": abs(sentiment_score - 0.5) > 0.3,
                    "retail_sentiment": prediction.direction
                }
                blackboard.update_insight("Sent", sent_insights)
            
            elif self.agent_id == "Fund":
                # 基本面分析师更新基本面洞察
                fund_insights = {
                    "confidence": prediction.confidence,
                    "reasoning": prediction.reasoning
                }
                blackboard.update_insight("Fund", fund_insights)
                
        except Exception as e:
            logger.error(f"❌ 更新洞察公告板失败: {e}")