import asyncio
import json
import logging
import pandas as pd
import os
import yaml  # 导入 yaml
from tqdm import tqdm
from typing import List, Dict, Any

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
        
        semaphore = asyncio.Semaphore(5) 
        
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
                "current_price": curr_row["Close"],
                "market_env": ctx.market_env if hasattr(ctx, "market_env") else "未知",
                "volatility": technical_indicators.get("volatility", 0),
                "timestamp": curr_date
            }

            @async_retry(retries=3, delay=1.0)
            async def ask_agent_safe(agent: HybridAgent) -> any:
                async with semaphore: 
                    return await agent.decide(ctx, chart_path=current_chart_input, market_data=market_data)

            tasks = [ask_agent_safe(a) for a in self.agents]
            actions = await asyncio.gather(*tasks)
            
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
            
            async def reflect_safe(pair: tuple) -> None:
                agent, action = pair
                async with semaphore:
                    await agent.reflect(ctx, action.direction, outcome, actual_chg, action.confidence, action.stake, market_data=market_data)

            await asyncio.gather(*[reflect_safe(p) for p in zip(self.agents, actions)])
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
        state_path = os.path.join(self.output_dir, 'backtest_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f)
        logger.info(f"💾 保存回测状态到: {state_path} (索引: {current_index})")

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