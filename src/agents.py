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

# 健壮的 JSON 解析器
def robust_json_parser(raw_text):
    try:
        # 尝试直接解析
        return json.loads(raw_text)
    except:
        # 尝试正则提取第一个 { 到 最后一个 }
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None # 解析彻底失败才返回 None

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
        if role == "Technical_Analyst" or role == "Sentiment_Watcher":
            try:
                # 尝试使用本地模型
                self.model_type = "local_vision"
                # 不同角色使用不同模型
                if role == "Technical_Analyst":
                    # 技术分析师使用视觉模型
                    model_name = "qwen3-vl:8b"
                    logger.info(f"🤖 初始化 {role} 代理，使用模型: {model_name} (视觉模型，强制JSON模式)")
                else:  # Sentiment_Watcher
                    # 情绪分析师使用纯文本模型
                    model_name = "qwen3:8b"
                    logger.info(f"🤖 初始化 {role} 代理，使用模型: {model_name} (纯文本模型，强制JSON模式)")
                
                # 提高准确率的配置
                self.llm = ChatOllama(
                    model=model_name,  # 使用对应的模型
                    temperature=0.3 if role == "Technical_Analyst" else 0.5, 
                    num_ctx=8192,  # 增大上下文窗口以提高理解能力
                    num_gpu=-1,  # 使用所有可用GPU
                    num_batch=4,  # 增加批处理大小以提高性能
                    f16_kv=False,  # 使用FP32键值缓存以提高准确性
                    logits_all=True,  # 输出所有token的logits以提高生成质量
                    use_mmap=True,  # 使用内存映射
                    use_mlock=False,  # 不锁定内存
                    keep_alive="12h",  # 保持时间
                    top_k=60,  # 增加候选词数量以提高生成质量
                    top_p=0.95,  # 调整核采样参数以提高多样性和准确性
                    repeat_penalty=1.1,  # 重复惩罚
                    num_thread=16,  # 增加CPU线程数以提高性能
                    format="json"  # 开启强制JSON模式
                )
            except Exception as e:
                # 如果本地模型不可用，切换到云模型
                logger.warning(f"⚠️  本地模型初始化失败: {e}，切换到云模型")
                self.model_type = "cloud_reasoning"
                self.llm = ChatOpenAI(
                    model="Pro/deepseek-ai/DeepSeek-V3.2", 
                    api_key="sk-xbobpvwcbzocotlefnqkgjusmxpejoeguqynqymzeauncwyc", 
                    base_url="https://api.siliconflow.cn/v1", 
                    temperature=0.3 if role == "Technical_Analyst" else 0.5
                )
                logger.info(f"🤖 初始化 {role} 代理，使用云模型: DeepSeek-V3.2")
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
        similar_mems = self.vector_mem.find_similar_memories(query, top_k=5)  # 增加检索数量以提高准确率
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

        # 处理波动率数据，防止 NaN
        import numpy as np
        vol_val = context.volatility if hasattr(context, 'volatility') else 0.012
        if isinstance(vol_val, (float, int)):
            if np.isnan(vol_val) or np.isinf(vol_val):
                vol_val = 0.012
        else:
            vol_val = 0.012
        
        # 更新 context 的波动率值
        if hasattr(context, 'volatility'):
            context.volatility = vol_val
        
        # 确保 final_prompt 中的波动率数据已经被正确处理
        # 即使使用视觉模型，也需要确保提示词中的数据是正确的
        logger.info(f"✅ 波动率数据处理完成: {vol_val}")
        
        # 只有技术分析师（使用视觉模型）才处理图像
        if self.model_type == "local_vision" and chart_path and self.role == "Technical_Analyst":
            # 检查图表路径是否存在
            if isinstance(chart_path, str) and not os.path.exists(chart_path):
                logger.warning(f"⚠️  图表文件不存在: {chart_path}")
            else:
                # 优化图像处理，控制图像大小在600×600左右
                img_b64 = self._encode_image(chart_path, max_size=600)  # 控制图像大小
                if img_b64:
                    logger.info(f"✅ 成功编码图表，大小: {len(img_b64) // 1024}KB")
                    # 确保提示词中的数据是正确的
                    # 为技术分析师创建一个包含正确波动率数据的提示词
                    tech_prompt = final_prompt
                    # 确保技术分析师的提示词中包含正确的波动率数据
                    logger.info(f"📊 技术分析师使用的波动率数据: {vol_val}")
                    message = [HumanMessage(content=[
                        {"type": "text", "text": tech_prompt + f"\n\n当前波动率: {vol_val*100:.2f}%\nPlease analyze the chart below:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}  
                    ])]
                else:
                    logger.warning(f"⚠️  图表编码失败")
                    message = final_prompt
        else:
            # 纯文本模型，不处理图像
            message = final_prompt

        try:
            # 执行模型推理
            res = await self.llm.ainvoke(message)
            content = self._clean_think_tag(res.content)
            
            # 添加调试日志
            logger.info(f"模型原始输出 (前500字符): {content[:500]}")
            
            try:
                # 增强的JSON提取逻辑，使用健壮的解析器
                logger.info(f"模型原始输出完整内容: {content}")
                
                # 1. 清理内容，移除可能的Markdown标记
                cleaned_content = content.replace('```json', '').replace('```', '').strip()
                
                # 2. 使用健壮的JSON解析器
                data = robust_json_parser(cleaned_content)
                
                if data:
                    logger.info(f"成功解析JSON: {json.dumps(data)[:200]}...")
                else:
                    # 尝试从整个内容中提取关键字段
                    logger.info("尝试从内容中提取关键字段")
                    # 提取方向
                    direction_match = re.search(r'direction\s*:\s*["\']?([A-Z]+)["\']?', cleaned_content, re.IGNORECASE)
                    direction = direction_match.group(1) if direction_match else "HOLD"
                    # 提取信心值
                    confidence_match = re.search(r'confidence\s*:\s*([\d.]+)', cleaned_content)
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.4
                    # 提取推理
                    reasoning_match = re.search(r'reasoning\s*:\s*["\']?([^"\']+)["\']?', cleaned_content, re.DOTALL)
                    reasoning = reasoning_match.group(1) if reasoning_match else "No JSON found"
                    # 构建数据
                    data = {"direction": direction, "confidence": confidence, "reasoning": reasoning}
                    logger.info(f"从内容中提取关键字段成功: {data}")
            except Exception as e:
                logger.error(f"JSON解析失败: {e}")
                # 即使解析失败，也要设置一个合理的默认值，而不是0
                data = {"direction": "HOLD", "confidence": 0.4, "reasoning": f"Parse Error: {str(e)[:30]}"}
                logger.info(f"解析失败后设置默认值: {data}")

            # 创建预测
            prediction = Prediction(
                agent_id=self.agent_id,
                direction=str(data.get("direction", "HOLD")).upper(),
                confidence=self._safe_float(data.get("confidence"), 0.5),
                stake=self._safe_float(data.get("stake"), 0.0),
                reasoning=data.get("reasoning", "")[:300],  # 增加推理长度以提高准确率
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
            
            # F. 清理显存
            if self.model_type == "local_vision":
                logger.info(f"🧹 清理 {self.agent_id} 的显存占用")
                # 强制垃圾回收
                import gc
                gc.collect()
                # 短暂休眠，确保显存释放
                await asyncio.sleep(1.0)
            
            return prediction
        except Exception as e:
            logger.error(f"Agent {self.agent_id} decide error: {e}")
            error_prediction = Prediction(self.agent_id, "HOLD", 0.0, 0.0, f"Error: {str(e)[:50]}")
            # 更新公告板记录错误
            self._update_blackboard(error_prediction, context, None)
            
            # 清理显存
            if self.model_type == "local_vision":
                logger.info(f"🧹 清理 {self.agent_id} 的显存占用")
                import gc
                gc.collect()
                await asyncio.sleep(1.0)
            
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

    def _encode_image(self, chart_source, max_size=600):
        try:
            import PIL
            from PIL import Image
            import io
            
            if isinstance(chart_source, io.BytesIO): 
                # 读取图像并调整大小
                image = Image.open(chart_source)
                # 调整图像大小
                image.thumbnail((max_size, max_size), PIL.Image.LANCZOS)
                # 保存到新的BytesIO
                resized_buffer = io.BytesIO()
                image.save(resized_buffer, format='PNG')
                resized_buffer.seek(0)
                return base64.b64encode(resized_buffer.getvalue()).decode("utf-8")
            elif isinstance(chart_source, str):
                # 检查文件是否存在
                if os.path.exists(chart_source):
                    # 检查文件大小，确保不是空文件
                    if os.path.getsize(chart_source) > 0:
                        # 读取图像并调整大小
                        image = Image.open(chart_source)
                        # 调整图像大小
                        image.thumbnail((max_size, max_size), PIL.Image.LANCZOS)
                        # 保存到新的BytesIO
                        resized_buffer = io.BytesIO()
                        image.save(resized_buffer, format='PNG')
                        resized_buffer.seek(0)
                        return base64.b64encode(resized_buffer.getvalue()).decode("utf-8")
                    else:
                        logger.warning(f"⚠️  图片文件为空: {chart_source}")
                else:
                    logger.warning(f"⚠️  图片文件不存在: {chart_source}")
        except Exception as e:
            logger.error(f"❌ 图片编码失败: {e}")
            # 如果调整大小失败，尝试直接读取
            try:
                if isinstance(chart_source, io.BytesIO):
                    return base64.b64encode(chart_source.getvalue()).decode("utf-8")
                elif isinstance(chart_source, str) and os.path.exists(chart_source) and os.path.getsize(chart_source) > 0:
                    with open(chart_source, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
            except Exception as e2:
                logger.error(f"❌ 直接读取图片也失败: {e2}")
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
            # 获取当前时间戳
            timestamp = context.timestamp if hasattr(context, 'timestamp') else None
            
            # 根据Agent角色更新不同类型的洞察
            if self.agent_id == "Tech":
                # 技术分析师更新技术层面洞察
                tech_insights = {
                    "timestamp": timestamp,  # 加入时间戳
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
                    "timestamp": timestamp,  # 加入时间戳
                    "bias": prediction.direction,
                    "confidence": prediction.confidence,
                    "factors": [prediction.reasoning[:100]]
                }
                blackboard.update_insight("Macro", macro_insights)
            
            elif self.agent_id == "Risk":
                # 风险管理者更新风险管理洞察
                risk_insights = {
                    "timestamp": timestamp,  # 加入时间戳
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
                    "timestamp": timestamp,  # 加入时间戳
                    "sentiment_score": sentiment_score,
                    "sentiment_extreme": abs(sentiment_score - 0.5) > 0.3,
                    "retail_sentiment": prediction.direction
                }
                blackboard.update_insight("Sent", sent_insights)
            
            elif self.agent_id == "Fund":
                # 基本面分析师更新基本面洞察
                fund_insights = {
                    "timestamp": timestamp,  # 加入时间戳
                    "confidence": prediction.confidence,
                    "reasoning": prediction.reasoning
                }
                blackboard.update_insight("Fund", fund_insights)
                
        except Exception as e:
            logger.error(f"❌ 更新洞察公告板失败: {e}")