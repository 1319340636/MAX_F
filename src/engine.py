# src/engine.py
import asyncio
import logging
import pandas as pd
from tqdm import tqdm
from typing import List

from src.utils import build_market_context, async_retry
from src.agents import HybridAgent
from src.decision_fusion import DecisionFusionEngine
from src.mechanism import PredictionMarket
from src.reporting import save_comprehensive_report
from src.visualization import plot_backtest_results
from src.data_loader import BacktestConfig
from src.visualization import ChartGenerator, plot_backtest_results

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config: BacktestConfig, df: pd.DataFrame):
        self.cfg = config
        self.df = df
        self.market = PredictionMarket()
        self.agents = self._init_agents()
        
        self.portfolio = [self.cfg.initial_capital]
        self.benchmark = [self.cfg.initial_capital]
        self.dates = [self.df.index[0]]
        self.agent_votes_history = [{}]
        self.memories = [{"total_mem_len": 0}]
        self.output_dir = self.cfg.report_dir
        
        # 初始化决策融合
        agent_ids = [a.agent_id for a in self.agents]
        self.fusion_engine = DecisionFusionEngine(agent_ids)
        
    def _init_agents(self) -> List[HybridAgent]:
        return [
            HybridAgent("Tech", "Technical_Analyst", self.cfg.ticker),
            HybridAgent("Sent", "Sentiment_Watcher", self.cfg.ticker),
            HybridAgent("Risk", "Risk_Manager", self.cfg.ticker),
            HybridAgent("Macro", "Macro_Economist", self.cfg.ticker),
            HybridAgent("Fund", "Fundamental_Analyst", self.cfg.ticker),
        ]

    async def run(self):
        logger.info(f"🚀 启动回测 (Async Engine): {self.cfg.symbol_name}")
        initial_price = self.df.iloc[0]["Close"]
        
        # 信号量控制并发
        semaphore = asyncio.Semaphore(self.cfg.max_workers)
        
        for i in tqdm(range(len(self.df) - 1), desc="📅 模拟交易", leave=True):
            curr_date, next_date = self.df.index[i], self.df.index[i+1]
            curr_row = self.df.loc[curr_date]
            
            chart_bytes = ChartGenerator.get_chart_bytes(self.df, i) if i >= self.cfg.min_kline_count - 1 else None
            ctx = build_market_context(curr_row.to_dict(), curr_date, self.cfg.ticker, "1D")

            if ctx is None:
                self._skip_day(next_date)
                continue

            # 1. 异步决策
            @async_retry(retries=3, delay=1.0)
            async def ask_agent_safe(agent):
                async with semaphore: 
                    return await agent.decide(ctx, chart_path=chart_bytes)

            tasks = [ask_agent_safe(a) for a in self.agents]
            actions = await asyncio.gather(*tasks)

            self._record_votes(curr_date, actions)

            # 2. 融合决策
            fusion_result = self.fusion_engine.get_weighted_decision(actions)
            sys_dir = fusion_result["direction"]
            pos_ratio = fusion_result["stake"]

            # 3. 市场揭晓
            actual_chg = (self.df.loc[next_date]["Close"] - curr_row["Close"]) / curr_row["Close"]
            outcome = "LONG" if actual_chg > 0 else "SHORT"
            
            if sys_dir == "HOLD":
                pnl = 0.0
            else:
                raw_pnl = actual_chg if sys_dir == "LONG" else -actual_chg
                pnl = (raw_pnl - self.cfg.total_cost_rate) * pos_ratio

            # 4. 闭环学习
            self.portfolio.append(round(self.portfolio[-1] * (1 + pnl), 4))
            self.benchmark.append(round((self.df.loc[next_date]["Close"] / initial_price) * self.cfg.initial_capital, 4))
            self.dates.append(next_date)

            self.market.resolve_market(actions, outcome) 
            
            async def reflect_safe(agent_action_pair):
                agent, action = agent_action_pair
                async with semaphore:
                    await agent.reflect(ctx, action.direction, outcome, actual_chg, action.confidence, action.stake)

            reflect_tasks = [reflect_safe(pair) for pair in zip(self.agents, actions)]
            await asyncio.gather(*reflect_tasks)

            self.fusion_engine.update_performance(actions, outcome)
            self._record_memory_stats()
            if chart_bytes: chart_bytes.close()

        self._finalize()
        self._auto_plot()
        logger.info(f"🚀 流程结束！结果已保存在: {self.output_dir}")
        
    def _skip_day(self, next_date):
        for arr in [self.portfolio, self.benchmark]: arr.append(arr[-1])
        self.dates.append(next_date)
        self.agent_votes_history.append({})
        self.memories.append(self.memories[-1] if self.memories else {"total_mem_len": 0})

    def _record_votes(self, curr_date, actions):
        d = {"date": curr_date.strftime("%Y-%m-%d")}
        for a in actions: d.update({f"{a.agent_id}_dir": a.direction, f"{a.agent_id}_conf": a.confidence})
        self.agent_votes_history.append(d)

    def _record_memory_stats(self):
        # 兼容性处理
        try:
            # 尝试获取 episodic_memory 的长度
            m = {f"{a.agent_id}_mem": len(a.memory_system.episodic_memory) for a in self.agents}
        except:
            m = {f"{a.agent_id}_mem": 0 for a in self.agents}
        m["total_mem_len"] = sum(m.values())
        self.memories.append(m)

    def _finalize(self):
        save_comprehensive_report(self.dates, self.portfolio, self.benchmark, self.agent_votes_history, self.memories, self.cfg.symbol_name, self.cfg.report_dir)

    def _auto_plot(self):
        print("\n📊 正在生成可视化分析报告...")
        try:
            plot_backtest_results(self.output_dir)
        except Exception as e:
            logger.error(f"❌ 绘图失败: {e}")