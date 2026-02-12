# MAX_F 系统 - AI 量化交易系统

## 📋 项目简介

MAX_F 是一个基于人工智能的量化交易系统，集成了现代AI技术（如LangChain）和传统量化交易策略，采用多角色决策机制，为不同资产类别（黄金、加密货币等）提供智能化的交易决策。

### 核心特点

- **模块化架构**：清晰的代码组织结构，便于维护和扩展
- **AI驱动决策**：集成LangChain和大语言模型，实现智能决策
- **多角色协同**：风险管理、技术分析、宏观经济分析等多角色协作
- **配置驱动**：YAML配置文件，灵活调整交易策略参数
- **记忆驱动**：实现交易记忆和模式识别
- **多资产支持**：支持黄金、加密货币等多种交易品种

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 必要的Python包（见requirements.txt）

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <项目地址>
   cd MAX_F-master
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置API密钥**
   - 如果使用OpenAI模型，需要设置环境变量：
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key
   
   # Linux/Mac
   export OPENAI_API_KEY=your_api_key
   ```

4. **运行回测**
   ```bash
   python run_backtest.py
   ```

## 📁 项目结构

```
MAX_F-master/
├── config/               # 配置文件目录
│   ├── trade_config.yaml       # 默认交易配置
│   └── trade_config_gold.yaml  # 黄金专用配置
├── data/                 # 历史数据目录
│   ├── BTC-USD_2022-01-01_2023-01-01.csv
│   └── BTC-USD_2023-01-01_2023-12-31.csv
├── data_cache/           # 缓存数据目录
├── logs/                 # 日志和图表目录
├── memory_store/         # 记忆存储目录
│   ├── Fund/
│   ├── Macro/
│   ├── Risk/
│   ├── Sent/
│   └── Tech/
├── src/                  # 源代码目录
│   ├── agents.py         # AI 代理模块
│   ├── data_loader.py    # 数据加载模块
│   ├── decision_fusion.py # 决策融合模块
│   ├── engine.py         # 回测引擎
│   ├── mechanism.py      # 交易机制
│   ├── memory.py         # 记忆管理模块
│   ├── models.py         # 数据模型
│   ├── prompts.py        # 提示词模板
│   ├── reporting.py      # 报告生成模块
│   ├── utils.py          # 工具函数
│   └── visualization.py  # 可视化模块
├── run_backtest.py       # 主运行脚本
├── requirements.txt      # 依赖文件
└── README.md             # 项目文档
```

## ⚙️ 配置说明

### 交易配置文件

配置文件位于 `config/` 目录，使用YAML格式：

- **trade_config.yaml**：默认通用配置
- **trade_config_gold.yaml**：黄金专用配置

### 主要配置参数

```yaml
# 市场类型配置
market_types:
  gold:
    # 关键词匹配
    keywords: ["XAU", "GC=F", "GOLD", "黄金"]
    
    # 风险管理参数
    stop_loss_pct: 0.015        # 1.5% 止损
    take_profit_pct: 0.035      # 3.5% 止盈
    single_position_pct: 0.15   # 单笔仓位上限
    total_position_pct: 0.30    # 总仓位上限
    
    # 角色定义和策略
    roles:
      Risk_Manager: >
        # 风险管理策略
      Technical_Analyst: >
        # 技术分析策略
      Macro_Economist: >
        # 宏观经济分析策略
```

## 📊 使用方法

### 运行回测

1. **启动系统**
   ```bash
   python run_backtest.py
   ```

2. **选择资产类别**
   - 系统会自动检测资产类型并加载相应的配置
   - 支持的资产：黄金（GOLD）、加密货币（BTC、ETH）

3. **查看回测结果**
   - 回测结果会保存在 `logs/` 目录
   - 包含详细的交易记录和图表

### 自定义策略

1. **修改配置文件**
   - 编辑 `config/trade_config.yaml` 调整策略参数
   - 或创建新的配置文件用于特定资产

2. **添加新资产支持**
   - 在配置文件中添加新的市场类型配置
   - 在 `run_backtest.py` 中添加资产识别逻辑

## 🧪 测试

项目包含基础测试框架，位于 `tests/` 目录：

```bash
# 运行测试
pytest tests/
```

## 🛠️ 开发指南

### 代码风格

- 遵循PEP 8代码风格
- 使用类型注解
- 保持函数和类的简洁性

### 贡献流程

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 运行测试
5. 发起Pull Request

## 📞 支持

### 问题反馈

- **Bug报告**：创建GitHub Issue
- **功能请求**：创建GitHub Issue
- **代码贡献**：提交Pull Request

### 联系方式

- 项目维护者：[您的名字]
- 邮箱：[您的邮箱]

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 📋 版本历史

- **v1.0.0** (2026-02-11)：初始版本
  - 支持黄金和加密货币交易
  - 集成LangChain AI决策
  - 多角色协同系统
  - 基础回测功能

---

*"智能交易，赢在未来"* 🌟
