# ==========================================
# 文件名: src/prompts.py
# 功能: Prompt 模板管理 (内置配置加载，无需依赖外部 config)
# ==========================================

import yaml
import os
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# 0. 内置配置加载逻辑 (解决 ImportError)
# ==============================================================================
def _load_local_config():
    """
    仅在当前文件内部使用的配置加载器。
    尝试寻找 config/trade_config.yaml 并加载。
    """
    # 尝试多种路径，兼容根目录运行和 src 目录运行
    possible_paths = [
        "config/trade_config.yaml",
        "../config/trade_config.yaml",
        os.path.join(os.getcwd(), "config/trade_config.yaml")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"❌ Prompts 加载配置失败: {e}")
                return {}
    
    logger.warning("⚠️ Prompts 未找到配置文件，使用默认空配置。")
    return {}

# 加载配置到当前模块变量
TRADE_CONFIG = _load_local_config()

def get_config_for_symbol(symbol: str):
    """
    根据品种自动匹配配置 (Gold vs Crypto)
    """
    market_types = TRADE_CONFIG.get("market_types", {})
    if not market_types: # 如果没配置，返回空
        return {}, "default"

    symbol = symbol.upper()
    for m_type, data in market_types.items():
        # 兼容 data 可能没有 keywords 的情况
        keywords = data.get("keywords", [])
        for kw in keywords:
            if kw in symbol:
                return data, m_type
    
    # 默认兜底：如果没有 crypto，就返回空字典
    return market_types.get("crypto", {}), "crypto"

# ==============================================================================
# 🎯 动态角色生成 (融合了 YAML 原则)
# ==============================================================================
def get_role_system_prompt(role: str, context, lessons_text: str = "(暂无)") -> str:
    """
    终极版 Prompt：注入 YAML 中的原则库 (Principles) 和 动态规则 (Dynamic Adjustment)
    """
    conf, m_type = get_config_for_symbol(context.symbol)
    
    # 1. 获取基础配置
    roles_cfg = conf.get("roles", {})
    # 如果特定品种没配角色，去全局配里找
    global_roles = TRADE_CONFIG.get("roles", {})
    
    role_instruction = roles_cfg.get(role, "")
    if not role_instruction:
        role_instruction = global_roles.get(role, "专业金融分析师，基于数据进行客观分析。")
    
    # 2. 提取硬参数 (带默认值兜底)
    sl_pct = conf.get("stop_loss_pct", 0.015) * 100
    tp_pct = conf.get("take_profit_pct", 0.035) * 100
    pos_limit = conf.get("single_position_pct", 0.15) * 100
    total_limit = conf.get("total_position_pct", 0.30) * 100
    vol_trigger = conf.get("flashbulb_vol", 1.5)

    # 3. 🔥 动态加载原则库 (Principles)
    principles_cfg = TRADE_CONFIG.get("principles", {})
    common_p = "\n- ".join(principles_cfg.get("common", ["风险控制第一"]))
    causal_p = "\n- ".join(principles_cfg.get("causal_logic", ["有因才有果"]))
    trend_p = "\n- ".join(principles_cfg.get("trend_trading", ["顺势而为"]))
    
    # 4. 🔥 加载动态调整规则 (作为参考建议)
    dynamic_cfg = TRADE_CONFIG.get("dynamic_adjustment", {})
    perf_adj = "\n- ".join(dynamic_cfg.get("performance_based_adjustment", []))
    market_adj = "\n- ".join(dynamic_cfg.get("market_adaptive", []))

    base_info = f"""
# 1. 角色设定 (Role: {role})
你是 MAS 系统中的核心成员。你的最高指令是：{role_instruction}

# 2. 核心参考：历史教训 (博学模式 - Working Memory)
⚠️ 必须优先对比当前行情与以下历史记录的相似性：
{lessons_text}

# 3. 市场实时数据 ({context.symbol})
- 当前价格: {context.price}
- 波动率: {context.volatility:.2%} (⚠️ 触发扩容阈值: {vol_trigger}%)
- 市场状态: {context.market_env}
- 当前账户持仓: {context.position_size} 手 (成本: {context.avg_cost})

# 4. 交易原则与动态调整 (宪法)
## A. 通用原则
- {common_p}
- {causal_p}

## B. 策略原则 (趋势/形态)
- {trend_p}

## C. 动态调整参考 (请根据近期胜率自我调节)
- {perf_adj}
- {market_adj}

# 5. 硬性风控红线 (Hard Limits)
1. 止损: 严格执行 {sl_pct}% 止损。
2. 止盈: 目标收益 {tp_pct}% 以上。
3. 仓位: 单笔上限 {pos_limit}%，总持仓上限 {total_limit}%。
"""
    return base_info.strip()

# ==============================================================================
# 📊 决策任务模板 (含量化标准 + 止损止盈)
# ==============================================================================
DECISION_TASK_TEMPLATE = """
{role_prompt}

[📜 核心原则]:
{principles_text}

[⚡ 历史教训]:
{lessons_text}

【当前任务】
基于上述信息预测未来24小时走势。

【⚠️ 量化判定标准 (必须严格执行)】
1. 置信度 (confidence):
   - 0.3~0.5: 仅单一技术面信号
   - 0.6~0.7: 技术面 + 宏观/情绪面共振
   - 0.8~1.0: 三维以上共振 (技术+宏观+基本面)
2. 仓位建议 (stake):
   - 震荡市: 150~200 (高抛低吸)
   - 趋势市: 100~150 (顺势)
   - 极端行情/波动率极高: 0~50 (防守)

【输出格式】
必须输出纯粹的JSON格式，包含具体的止损止盈位，严禁包含任何Markdown格式或额外文字：
{{
    "reasoning": "详细逻辑...",
    "direction": "LONG/SHORT/HOLD",
    "confidence": 0.8,
    "stake": 100.0,
    "position_ratio": 0.2,
    "stop_loss": 2010.5,
    "take_profit": 2080.0
}}
"""

# ==============================================================================
# 🧠 反思任务模板 (标签化)
# ==============================================================================
REFLECTION_TASK_TEMPLATE = """
我是{role}。针对 {symbol} ({market_env})。
预测: {my_action} | 实际: {actual_outcome} | 波动: {actual_return:.2%}

请总结一条改进教训，并严格按照以下格式：
1. 必须打上标签：【逻辑】或【方向】或【仓位】或【时机】
2. 字数限制 20 字以内。
3. 不要输出 <think> 标签或其他废话。

输出示例：
【仓位】波动率过大时应主动降杠杆。
【逻辑】忽略了美联储会议纪要的影响。
"""

# ==============================================================================
# 🤝 多智能体融合模板 (MAS 接口) - 可选
# ==============================================================================
FUSION_TASK_TEMPLATE = """
你是首席投资官(CIO)。负责融合 5 个分析师的意见。

当前品种: {symbol}
当前价格: {price}

【团队报告】
{role_decisions}

【决策逻辑】
1. 一票否决：若风控经理 (Risk_Manager) 建议空仓/止损，必须降低总仓位。
2. 多数原则：3个以上角色方向一致才可开单。
3. 止损取值：取所有 LONG 单的最高止损价，或 SHORT 单的最低止损价 (保守策略)。

【最终指令】
请输出最终合并后的 JSON 决策 (格式同上)。
"""