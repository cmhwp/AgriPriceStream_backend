# AgriPriceStream 蔬菜价格数据采集与分析系统

一个基于FastAPI的蔬菜价格数据采集、分析和预测系统，提供全面的蔬菜价格监控、历史分析和科学预测功能。

## 系统功能

- **数据采集**：从江南市场API爬取实时蔬菜价格数据，支持手动和自动化定时采集
- **数据分析**：
  - 历史价格查询和多维度趋势分析
  - 价格波动率和异常检测
  - 产地影响分析和溢价评估
  - 季节性价格变化规律挖掘
- **预测功能**：
  - 基于LSTM的价格预测模型
  - 个性化购买时机推荐
  - 多周期（短期、中期）预测
- **数据可视化**：
  - 价格走势图表
  - 产地分布词云
  - 价格波动热力图
  - 异常价格警报
- **系统管理**：全面的管理员数据管理和模型训练功能

## 技术栈

- **后端**：
  - **FastAPI** - 高性能Web框架
  - **SQLAlchemy** - ORM数据库操作
  - **MySQL** - 关系型数据库
  - **PyTorch** - 深度学习框架（LSTM价格预测）
  - **APScheduler** - 任务调度器
  - **JWT** - 身份认证
  - **Pydantic** - 数据验证

- **前端推荐**：
  - **React/Vue** - 前端框架
  - **ECharts** - 数据可视化图表
  - **ECharts-WordCloud** - 词云展示
  - **Ant Design/Element UI** - UI组件库

## 安装与配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置数据库

1. 创建MySQL数据库:

```sql
CREATE DATABASE agriconnect CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. 配置连接参数（app/db/database.py）:

```python
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://username:password@localhost:3306/agriconnect"
```

## 运行系统

```bash
uvicorn main:app --reload
```

访问系统API文档: http://localhost:8000/docs

## 数据分析功能

系统提供多维度的数据分析功能：

### 价格分析

- **基础统计**：平均价格、最高/最低价格、价格区间分析
- **波动分析**：价格波动率、单日价差、历史标准差
- **趋势分析**：日环比、周同比变化、长期价格走势
- **异常检测**：通过统计方法识别异常价格波动

### 产地分析

- **产地影响**：不同产地对蔬菜价格的影响评估
- **产地分布**：产地数量分布及占比，支持词云可视化
- **溢价分析**：高价值产地溢价空间评估

### 季节性分析

- **月度规律**：识别月度价格规律
- **季节特征**：挖掘季节性价格变动特征
- **最佳购买时机**：基于历史数据推荐最佳购买时机

## API 端点

系统包含以下主要API端点:

### 用户验证

- `POST /users/register` - 注册新用户
- `POST /users/token` - 获取访问令牌
- `GET /users/me` - 获取当前用户信息

### 蔬菜数据

- `GET /vegetables/` - 获取蔬菜列表（支持分页、过滤和排序）
- `GET /vegetables/{vegetable_id}` - 获取蔬菜详情
- `GET /vegetables/kinds` - 获取蔬菜种类列表
- `GET /vegetables/provenances` - 获取蔬菜产地列表
- `GET /vegetables/price-range` - 根据价格范围查询蔬菜
- `GET /vegetables/recent` - 获取最近更新的蔬菜

### 价格数据

- `GET /prices/records` - 获取价格记录
- `GET /prices/history` - 获取历史价格数据
- `GET /prices/chart-data` - 生成图表数据
- `GET /prices/real-time` - 获取实时价格
- `GET /prices/seasonality` - 分析价格季节性
- `GET /prices/three-day` - 最近三天价格对比

### 价格预测

- `GET /predictions/results/{vegetable_id}` - 获取价格预测结果
- `GET /predictions/best-purchase-day/{vegetable_id}` - 获取最佳购买日推荐
- `POST /predictions/train-model/{vegetable_id}` - 训练预测模型(管理员)
- `GET /predictions/training-history` - 获取模型训练历史

### 仪表盘

- `GET /dashboard` - 用户仪表盘数据
  - 包含丰富的数据统计和可视化资源：
    - 价格分析（基础统计、趋势、波动、异常）
    - 产地分析（词云数据）
    - 蔬菜种类统计
    - 热门蔬菜分析
    - 价格波动明细

### 管理员功能

- `GET /admin/dashboard` - 获取管理员仪表盘
- `POST /admin/crawl/run-once` - 执行一次爬虫任务
- `POST /admin/crawl/historical` - 爬取历史数据
- `POST /admin/users/{user_id}` - 更新用户信息
- `DELETE /admin/users/{user_id}` - 删除用户
- `GET /admin/crawler/status` - 爬虫状态
- `GET /admin/crawler/activities` - 爬虫活动记录

## 前端数据展示指南

### 产地词云展示

系统提供了专为词云优化的产地数据格式：

```json
{
  "provenances": [
    {
      "text": "山东",      // 产地名称，用于词云文本
      "value": 150,       // 产品数量，用于词云大小
      "weight": 100,      // 相对权重（1-100范围）
      "count": 150        // 原始数量（兼容旧代码）
    },
    ...
  ]
}
```

可使用ECharts-WordCloud等库进行可视化展示。

### 价格分析数据

Dashboard API 提供丰富的价格分析数据：

```json
{
  "price_analytics": {
    "date": "2023-08-15",
    "total_products": 120,
    "average_price": 5.23,
    "highest_price": 12.8,
    "lowest_price": 1.2,
    "avg_daily_spread": 1.5,
    "daily_trend": "up",
    "weekly_trend": "stable",
    "daily_change": 3.6,
    "weekly_change": 0.8,
    "price_volatility": [...],  // 价格波动最大的产品
    "premium_origins": [...],   // 价格溢价最高的产地
    "price_anomalies": [...]    // 价格异常的产品
  }
}
```

## 系统角色

系统支持两种用户角色:

1. **普通用户** - 可以查询价格、分析趋势、获取预测、使用多维度数据分析功能
2. **管理员** - 除普通用户功能外，还可以管理用户、控制爬虫、训练模型、修正数据、查看系统状态

## 项目目录结构

```
AgriPriceStream_backend/
├── app/
│   ├── crud/        # 数据库操作
│   │   ├── users.py        # 用户操作
│   │   ├── vegetables.py   # 蔬菜操作
│   │   ├── prices.py       # 价格操作
│   │   ├── predictions.py  # 预测操作
│   │   └── dashboard.py    # 仪表盘数据
│   ├── db/          # 数据库配置
│   ├── models/      # 数据模型
│   │   ├── models.py       # 数据库模型
│   │   └── saved/          # 保存的预测模型
│   ├── routers/     # API路由
│   ├── schemas/     # 数据模式
│   └── utils/       # 工具模块
│       ├── crawler.py       # 爬虫实现
│       ├── scheduler.py     # 定时任务
│       ├── prediction.py    # 预测模型
│       └── auth.py          # 身份验证
├── main.py          # 应用入口
└── requirements.txt # 项目依赖
```

## 系统架构

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│ 数据采集层  │ ──→ │  数据存储层  │ ──→ │  应用服务层  │
└─────────────┘      └──────────────┘      └──────────────┘
      │                                            │
      ↓                                            ↓
┌─────────────┐                          ┌──────────────┐
│ 爬虫调度器  │                          │   API 接口   │
└─────────────┘                          └──────────────┘
                                                │
                                                ↓
                                         ┌──────────────┐
                                         │  前端应用层  │
                                         └──────────────┘
```

## 后续开发计划

- 添加更多数据源支持
- 实现更精确的价格预测算法
- 优化数据采集效率
- 扩展更多可视化场景
- 增加区域价格比较分析

## 许可证

MIT 