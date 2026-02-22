import asyncio
import json
import logging
import pandas as pd
import os
import yaml  # 导入 yaml
from tqdm import tqdm
from typing import List, Dict, Any
from datetime import datetime

# --- 注意：这里删除了错误的 from src.engine import BacktestEngine ---

from src.utils import build_market_context, async_retry
from src.agents import HybridAgent
from src.decision_fusion import DecisionFusionEngine
from src.mechanism import PredictionMarket
from src.reporting import save_comprehensive_report
from src.data_loader import BacktestConfig
from src.visualization import ChartGenerator, plot_backtest_results

logger = logging.getLogger(__name__)

class BacktestEngine:
    # 增加 strategy_config_path 参数
    def __init__(self, config: BacktestConfig, df: pd.DataFrame, strategy_config_path: str = "config/trade_config.yaml"):
        self.cfg = config
        self.df = df
        
        # 加载指定的策略配置文件 (Gold 或 Crypto)
        self.strategy_config_path = strategy_config_path
        self.trade_config = self._load_strategy_config(strategy_config_path)
        
        self.market = PredictionMarket()
        self.agents = self._init_agents()
        self.portfolio = [self.cfg.initial_capital]
        self.benchmark = [self.cfg.initial_capital]
        self.dates = [self.df.index[0]]
        self.agent_votes_history = [{}]
        self.memories = [{"total_mem_len": 0}]
        self.output_dir = self.cfg.report_dir
        self.chart_dir = os.path.join(self.output_dir, "charts")
        agent_ids = [a.agent_id for a in self.agents]
        self.fusion_engine = DecisionFusionEngine(agent_ids)
        
    # 辅助函数用于安全加载 YAML
    def _load_strategy_config(self, path: str) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                logger.info(f"📄 引擎正在加载策略文件: {path}")
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"❌ 策略文件加载失败: {path}, 错误: {e}")
            raise e

    def _init_agents(self) -> List[HybridAgent]:
        # 1. 正常初始化 Agent
        agents = [
            HybridAgent("Tech", "Technical_Analyst", self.cfg.ticker),
            HybridAgent("Sent", "Sentiment_Watcher", self.cfg.ticker),
            HybridAgent("Risk", "Risk_Manager", self.cfg.ticker),
            HybridAgent("Macro", "Macro_Economist", self.cfg.ticker),
            HybridAgent("Fund", "Fundamental_Analyst", self.cfg.ticker),
        ]
        
        # 2. 将加载好的专用策略配置“注入”给所有 Agent
        for agent in agents:
            agent.config = self.trade_config 
            agent.strategy_config = self.trade_config
            
        return agents
    
    def _calculate_technical_indicators(self, index: int) -> Dict[str, Any]:
        """计算技术指标
        
        Args:
            index: 当前数据索引
            
        Returns:
            Dict: 技术指标
        """
        indicators = {}
        
        # 计算过去5天的最高价（用于突破验证）
        if index >= 4:
            high_n_days = self.df['High'].iloc[index-4:index+1].max()
            indicators['high_n_days'] = high_n_days
        
        # 计算简单移动平均线
        if index >= 19:
            ma20 = self.df['Close'].iloc[index-19:index+1].mean()
            indicators['ma20'] = ma20
        
        if index >= 59:
            ma60 = self.df['Close'].iloc[index-59:index+1].mean()
            indicators['ma60'] = ma60
        
        # 计算波动率
        if index >= 9:
            volatility = self.df['Close'].iloc[index-9:index+1].pct_change().std() * 100
            indicators['volatility'] = volatility
        
        # 计算阻力位和支撑位（简单版本）
        if index >= 19:
            recent_highs = self.df['High'].iloc[index-19:index+1]
            recent_lows = self.df['Low'].iloc[index-19:index+1]
            
            # 阻力位：最近高点的平均值
            resistance_level = recent_highs.nlargest(3).mean()
            indicators['resistance_level'] = resistance_level
            
            # 支撑位：最近低点的平均值
            support_level = recent_lows.nsmallest(3).mean()
            indicators['support_level'] = support_level
        
        return indicators

    async def run(self) -> None:
        logger.info(f"🚀 启动回测 (Async Engine): {self.cfg.symbol_name} | 策略: {os.path.basename(self.strategy_config_path)}")
        
        # 加载状态，检查是否需要断点续传
        state = self._load_state()
        start_index = 0
        # 始终定义初始价格
        initial_price = self.df.iloc[0]["Close"]
        
        if state:
            # 恢复状态
            self.portfolio = state['portfolio']
            self.benchmark = state['benchmark']
            self.dates = [pd.to_datetime(date) for date in state['dates']]
            self.agent_votes_history = state['agent_votes_history']
            self.memories = state['memories']
            start_index = state['current_index'] + 1
            logger.info(f"🔄 从索引 {start_index} 继续回测")
        
        # 利用多进程预绘图 (调用 visualization.py 里的新方法)
        ChartGenerator.precompute_charts(self.df, self.chart_dir, window=40, max_workers=24)
        
        semaphore = asyncio.Semaphore(4)  # 增加并发数，提高处理速度和准确率
        
        # 从 start_index 开始迭代
        for i in tqdm(range(start_index, len(self.df) - 1), desc="📅 模拟交易", leave=True):
            curr_date, next_date = self.df.index[i], self.df.index[i+1]
            curr_row = self.df.loc[curr_date]
            
            # 读取预生成的图片路径
            chart_path = os.path.join(self.chart_dir, f"chart_{i}.png")
            
            # 增强图片路径检查
            current_chart_input = None
            if i >= self.cfg.min_kline_count - 1:
                if os.path.exists(chart_path):
                    if os.path.getsize(chart_path) > 0:
                        current_chart_input = chart_path
                        logger.info(f"📊 使用图表: {chart_path}")
                    else:
                        logger.warning(f"⚠️  图表文件为空: {chart_path}")
                else:
                    logger.warning(f"⚠️  图表文件不存在: {chart_path}")
            ctx = build_market_context(curr_row.to_dict(), curr_date, self.cfg.ticker, "1D")

            if ctx is None:
                self._skip_day(next_date)
                continue

            # 计算技术指标，构造market_data
            technical_indicators = self._calculate_technical_indicators(i)
            market_data = {
            **technical_indicators,
            "context": ctx.__dict__ if hasattr(ctx, "__dict__") else {},
            "current_price": round(curr_row["Close"], 2),  # 四舍五入到小数点后两位
            "market_env": ctx.market_env if hasattr(ctx, "market_env") else "未知",
            "volatility": round(technical_indicators.get("volatility", 0), 4),  # 波动率保留四位小数
            "timestamp": curr_date
        }

            # 分离 qwen 代理和 deepseek 代理
            qwen_agents = [agent for agent in self.agents if agent.role in ["Technical_Analyst", "Sentiment_Watcher"]]
            deepseek_agents = [agent for agent in self.agents if agent.role not in ["Technical_Analyst", "Sentiment_Watcher"]]
            
            # 并行执行 qwen 代理（技术分析师和情绪分析师）
            async def process_qwen_agent(agent):
                try:
                    action = await agent.decide(ctx, chart_path=current_chart_input, market_data=market_data)
                    # 强制垃圾回收，释放显存
                    import gc
                    gc.collect()
                    return action
                except Exception as e:
                    logger.error(f"处理 qwen 代理时出错: {e}")
                    # 返回默认动作
                    from src.models import Prediction
                    return Prediction(agent.agent_id, "HOLD", 0.3, 0.0, f"Error: {str(e)[:30]}")
            
            # 并行执行所有 qwen 代理
            qwen_tasks = [process_qwen_agent(agent) for agent in qwen_agents]
            qwen_actions = await asyncio.gather(*qwen_tasks)
            actions = qwen_actions
            
            # 并发执行 deepseek 代理（其他三个代理）
            if deepseek_agents:
                @async_retry(retries=3, delay=1.0)
                async def ask_deepseek_safe(agent: HybridAgent) -> any:
                    return await agent.decide(ctx, chart_path=current_chart_input, market_data=market_data)
                
                deepseek_tasks = [ask_deepseek_safe(a) for a in deepseek_agents]
                deepseek_actions = await asyncio.gather(*deepseek_tasks)
                actions.extend(deepseek_actions)
            
            # 专家辩论机制
            debate_record = self._execute_agent_debate(actions, market_data)
            
            # 显示AI决策过程摘要
            print(f"\n📋 [{curr_date.strftime('%Y-%m-%d')}] AI决策过程摘要:")
            for action in actions:
                # 提取关键推理信息（前100个字符）
                key_reasoning = action.reasoning[:100] + "..." if len(action.reasoning) > 100 else action.reasoning
                print(f"  🤖 {action.agent_id}: {action.direction} (信心: {action.confidence:.2f})")
                print(f"     💭 推理: {key_reasoning}")
            
            # 生成决策融合结果
            fusion_result = self.fusion_engine.get_weighted_decision(actions)
            # 计算综合信心度
            total_weight = sum(fusion_result['weights'].values()) if fusion_result['weights'] else 0
            avg_confidence = total_weight / len(fusion_result['weights']) if fusion_result['weights'] else 0.0
            print(f"  🎯 系统决策: {fusion_result['direction']} (信心: {avg_confidence:.2f})")
            print(f"  💰 资金分配: {fusion_result['stake']:.2f}")
            
            # 生成每日AI分析报告
            self._generate_daily_analysis(i, curr_date, actions, market_data)
            
            self._record_votes(curr_date, actions)
            # 使用前面已经计算过的 fusion_result
            sys_dir, pos_ratio = fusion_result["direction"], fusion_result["stake"]
            actual_chg = (self.df.loc[next_date]["Close"] - curr_row["Close"]) / curr_row["Close"]
            outcome = "LONG" if actual_chg > 0 else "SHORT"
            
            pnl = 0.0 if sys_dir == "HOLD" else (actual_chg if sys_dir == "LONG" else -actual_chg - self.cfg.total_cost_rate) * pos_ratio
            self.portfolio.append(round(self.portfolio[-1] * (1 + pnl), 4))
            self.benchmark.append(round((self.df.loc[next_date]["Close"] / initial_price) * self.cfg.initial_capital, 4))
            self.dates.append(next_date)
            self.market.resolve_market(actions, outcome) 
            
            # 分离 qwen 代理和 deepseek 代理的动作
            qwen_actions = [action for agent, action in zip(self.agents, actions) if agent.role in ["Technical_Analyst", "Sentiment_Watcher"]]
            qwen_pairs = [(agent, action) for agent, action in zip(self.agents, actions) if agent.role in ["Technical_Analyst", "Sentiment_Watcher"]]
            deepseek_pairs = [(agent, action) for agent, action in zip(self.agents, actions) if agent.role not in ["Technical_Analyst", "Sentiment_Watcher"]]
            
            # 顺序执行 qwen 代理的反思
            for agent, action in qwen_pairs:
                await agent.reflect(ctx, action.direction, outcome, actual_chg, action.confidence, action.stake, market_data=market_data)
                # 增加等待时间，确保显存得到充分释放
                await asyncio.sleep(1.0)
                # 强制垃圾回收
                import gc
                gc.collect()
            
            # 并发执行 deepseek 代理的反思
            if deepseek_pairs:
                async def reflect_deepseek_safe(pair: tuple) -> None:
                    agent, action = pair
                    await agent.reflect(ctx, action.direction, outcome, actual_chg, action.confidence, action.stake, market_data=market_data)
                
                await asyncio.gather(*[reflect_deepseek_safe(p) for p in deepseek_pairs])
            self.fusion_engine.update_performance(actions, outcome)
            self._record_memory_stats()
            
            # 每10次迭代保存一次状态，或者在最后一次迭代保存
            if (i + 1) % 10 == 0 or (i + 1) == len(self.df) - 2:
                self._save_state(i)

        self._finalize()
        self._auto_plot()
        
    def _skip_day(self, next_date: any) -> None:
        for arr in [self.portfolio, self.benchmark]: arr.append(arr[-1])
        self.dates.append(next_date)
        self.agent_votes_history.append({})
        self.memories.append(self.memories[-1] if self.memories else {"total_mem_len": 0})

    def _record_votes(self, curr_date: any, actions: list) -> None:
        d = {"date": curr_date.strftime("%Y-%m-%d")}
        for a in actions: d.update({f"{a.agent_id}_dir": a.direction, f"{a.agent_id}_conf": a.confidence})
        self.agent_votes_history.append(d)

    def _record_memory_stats(self) -> None:
        try:
            m = {f"{a.agent_id}_mem": len(a.memory_system.episodic_memory) for a in self.agents}
        except:
            m = {f"{a.agent_id}_mem": 0 for a in self.agents}
        m["total_mem_len"] = sum(m.values())
        self.memories.append(m)

    def _finalize(self) -> None:
        save_comprehensive_report(self.dates, self.portfolio, self.benchmark, self.agent_votes_history, self.memories, self.cfg.symbol_name, self.cfg.report_dir)

    def _execute_agent_debate(self, actions: list, market_data: dict) -> str:
        """
        执行专家辩论机制
        
        Args:
            actions: 所有代理的决策结果
            market_data: 市场数据
            
        Returns:
            str: 辩论记录
        """
        try:
            # 识别各个代理的动作和实例
            tech_action = next((a for a in actions if a.agent_id == "Tech"), None)
            sent_action = next((a for a in actions if a.agent_id == "Sent"), None)
            risk_action = next((a for a in actions if a.agent_id == "Risk"), None)
            
            # 找到对应的代理实例
            tech_agent = next((agent for agent in self.agents if agent.agent_id == "Tech"), None)
            sent_agent = next((agent for agent in self.agents if agent.agent_id == "Sent"), None)
            risk_agent = next((agent for agent in self.agents if agent.agent_id == "Risk"), None)
            
            debate_record = "# 专家辩论记录\n\n"
            
            # 技术分析师给出初步建议
            if tech_action:
                debate_record += f"## 1. 技术分析师 (Tech) 初步建议\n"
                debate_record += f"**方向**: {tech_action.direction}\n"
                debate_record += f"**信心**: {tech_action.confidence:.2f}\n"
                debate_record += f"**理由**: {tech_action.reasoning}\n\n"
            
            # 情绪分析师（波动率专家）对技术分析师的建议进行评论
            if tech_action and sent_action:
                debate_record += f"## 2. 情绪分析师 (Sent) 评论\n"
                debate_record += f"**技术面建议**: {tech_action.direction} (信心: {tech_action.confidence:.2f})\n"
                
                # 基于 GVZ 数据进行评论
                gvz = market_data.get('GVZ', 0)
                gvz_zscore = market_data.get('GVZ_ZScore', 0)
                gvz_quantile = market_data.get('GVZ_Quantile', 0)
                
                debate_record += f"**GVZ 数据**: {gvz:.2f} (Z-Score: {gvz_zscore:.2f}, 分位数: {gvz_quantile:.2f})\n"
                debate_record += f"**情绪面建议**: {sent_action.direction}\n"
                debate_record += f"**评论**: {sent_action.reasoning}\n\n"
            
            # 风险经理根据两者的辩论做出最终判断
            if risk_action:
                debate_record += f"## 3. 风险经理 (Risk) 终审\n"
                debate_record += f"**最终仓位建议**: {risk_action.position_size if hasattr(risk_action, 'position_size') else 'N/A'}\n"
                debate_record += f"**风险评估**: {risk_action.reasoning}\n\n"
            
            debate_record += f"## 4. 辩论结论\n"
            debate_record += "基于技术面分析和情绪面评估，系统将采用上述风险控制措施。\n"
            
            # 保存辩论记录
            debate_dir = os.path.join(self.output_dir, 'debates')
            os.makedirs(debate_dir, exist_ok=True)
            debate_file = os.path.join(debate_dir, f"debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            
            with open(debate_file, 'w', encoding='utf-8') as f:
                f.write(debate_record)
            
            logger.info(f"✅ 专家辩论记录已保存至: {debate_file}")
            
            # 每次迭代都存入向量记忆系统，提高学习能力和准确率
            # 将辩论记录存入向量记忆系统
            self._store_debate_in_vector_memory(tech_agent, sent_agent, risk_agent, debate_record, market_data)
            
            return debate_record
            
        except Exception as e:
            logger.error(f"❌ 执行专家辩论失败: {e}")
            return "# 专家辩论记录\n\n**错误**: 辩论执行失败\n"
    
    def _store_debate_in_vector_memory(self, tech_agent, sent_agent, risk_agent, debate_record, market_data):
        """
        将辩论记录存入向量记忆系统
        
        Args:
            tech_agent: 技术分析师代理实例
            sent_agent: 情绪分析师代理实例
            risk_agent: 风险经理代理实例
            debate_record: 辩论记录
            market_data: 市场数据
        """
        try:
            # 为每个代理创建适合的记忆内容
            gvz = market_data.get('GVZ', 0)
            gvz_zscore = market_data.get('GVZ_ZScore', 0)
            
            # 存入技术分析师的向量记忆
            if tech_agent and hasattr(tech_agent, 'vector_mem'):
                tech_memory = f"【技术分析】辩论记录：基于技术面的建议，考虑情绪面反馈（GVZ: {gvz:.2f}, Z-Score: {gvz_zscore:.2f}）"
                tech_agent.vector_mem.add_memory(tech_memory)
                tech_agent.vector_mem.save()
                logger.info(f"✅ 辩论记录已存入技术分析师的向量记忆")
            
            # 存入情绪分析师的向量记忆
            if sent_agent and hasattr(sent_agent, 'vector_mem'):
                sent_memory = f"【情绪分析】辩论记录：基于 GVZ 数据的情绪评估，对技术面建议的评论（GVZ: {gvz:.2f}, Z-Score: {gvz_zscore:.2f}）"
                sent_agent.vector_mem.add_memory(sent_memory)
                sent_agent.vector_mem.save()
                logger.info(f"✅ 辩论记录已存入情绪分析师的向量记忆")
            
            # 存入风险经理的向量记忆
            if risk_agent and hasattr(risk_agent, 'vector_mem'):
                risk_memory = f"【风险评估】辩论记录：基于技术面和情绪面的综合风险评估，最终仓位建议"
                risk_agent.vector_mem.add_memory(risk_memory)
                risk_agent.vector_mem.save()
                logger.info(f"✅ 辩论记录已存入风险经理的向量记忆")
                
        except Exception as e:
            logger.error(f"❌ 存储辩论记录到向量记忆失败: {e}")

    def _generate_daily_analysis(self, current_index: int, curr_date: pd.Timestamp, actions: list, market_data: dict) -> None:
        """
        生成每日AI分析报告
        """
        # 创建分析目录
        analysis_dir = os.path.join(self.output_dir, 'daily_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 生成分析报告
        analysis = {
            'date': str(curr_date),
            'market_data': {
                'current_price': market_data.get('current_price'),
                'volatility': market_data.get('volatility'),
                'market_env': market_data.get('market_env'),
                'technical_indicators': {
                    k: v for k, v in market_data.items() if k not in ['context', 'current_price', 'market_env', 'volatility', 'timestamp']
                }
            },
            'agent_analyses': []
        }
        
        # 收集每个智能体的分析
        for action in actions:
            agent_analysis = {
                'agent_id': action.agent_id,
                'direction': action.direction,
                'confidence': action.confidence,
                'reasoning': action.reasoning,
                'stake': action.stake,
                'stop_loss': action.stop_loss,
                'take_profit': action.take_profit,
                'position_ratio': action.position_ratio
            }
            analysis['agent_analyses'].append(agent_analysis)
        
        # 生成决策融合结果
        fusion_result = self.fusion_engine.get_weighted_decision(actions)
        analysis['fusion_result'] = fusion_result
        
        # 保存分析报告
        analysis_path = os.path.join(analysis_dir, f'analysis_{current_index}_{curr_date.strftime("%Y%m%d")}.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"📈 生成每日分析报告: {analysis_path}")

    def _save_state(self, current_index: int) -> None:
        """
        保存回测状态，用于断点续传
        """
        state = {
            'current_index': current_index,
            'portfolio': self.portfolio,
            'benchmark': self.benchmark,
            'dates': [str(date) for date in self.dates],
            'agent_votes_history': self.agent_votes_history,
            'memories': self.memories
        }
        
        # 1. 保存到当前输出目录（用于本次回测的状态管理）
        state_path = os.path.join(self.output_dir, 'backtest_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f)
        logger.info(f"💾 保存回测状态到: {state_path} (索引: {current_index})")
        
        # 2. 同时保存到固定的 checkpoint 目录（用于断点重连）
        # 提取资产名称（如从 "GOLD" 或 "BTC"）
        symbol_name = self.cfg.symbol_name.upper()
        checkpoint_dir = f"logs/{symbol_name}_checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_state_path = os.path.join(checkpoint_dir, 'backtest_state.json')
        with open(checkpoint_state_path, 'w') as f:
            json.dump(state, f)
        logger.info(f"🔄 同时更新断点文件到: {checkpoint_state_path}")

    def _load_state(self) -> dict:
        """
        加载回测状态，用于断点续传
        """
        state_path = os.path.join(self.output_dir, 'backtest_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            logger.info(f"📥 加载回测状态从: {state_path} (索引: {state['current_index']})")
            return state
        return None

    def _auto_plot(self) -> None:
        print("\n📊 正在生成可视化分析报告...")
        try: plot_backtest_results(self.output_dir)
        except Exception as e: logger.error(f"❌ 绘图失败: {e}")