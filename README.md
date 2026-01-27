# 📈 Portfolio Optimization with Time Series Forecasting

## 🎯 Project Overview
A data-driven portfolio management system for **Guide Me in Finance (GMF) Investments** that combines time series forecasting with Modern Portfolio Theory to optimize asset allocation and enhance investment strategies.

### **Business Context**
GMF Investments seeks to leverage AI/ML models to predict market trends and construct optimal portfolios. This project implements a complete pipeline from data extraction to backtesting, providing actionable insights for financial analysts.

---

## 📊 Project Structure
```bash
portfolio-optimization/
├── data/              # Raw & processed financial data
├── notebooks/         # EDA and analysis notebooks
├── src/               # Source code for models and optimization
├── scripts/           # Utility scripts and automation
├── tests/             # Unit tests
├── README.md          # Project documentation
└── requirements.txt   # Dependencies
```
---

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- Git

### **Installation**
1. Clone the repository:
```bash
git clone https://github.com/Jaki77/portfolio-optimization.git
cd portfolio-optimization
```
2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📋 Tasks Implementation
### **Task 1: Data Preprocessing & EDA**
**Objective**: Understand data characteristics and prepare for modeling
**Key Deliverables:**
    - Cleaned TSLA, BND, SPY data (2015-2026)
    - Stationarity analysis (ADF test)
    - Risk metrics (VaR, Sharpe Ratio)
    - 10+ insightful visualizations

### **Task 2: Time Series Forecasting Models**
**Models Implemented:**
    - ARIMA/SARIMA (classical statistical)
    - LSTM (deep learning)
**Evaluation:** MAE, RMSE, MAPE comparison
**Best Model:** Selected based on performance metrics

### **Task 3: Future Market Trends Forecasting**
**Horizon:** 6-12 month forecasts
**Output:** Confidence intervals and trend analysis
**Business Insights:** Identified opportunities and risks

### **Task 4: Portfolio Optimization**
**Methodology:** Modern Portfolio Theory (MPT)
**Tools:** PyPortfolioOpt, Efficient Frontier
**Output:** Optimal portfolio weights (TSLA, BND, SPY)
**Key Portfolios:** Maximum Sharpe, Minimum Volatility

### **Task 5: Strategy Backtesting**
**Benchmark:** 60% SPY / 40% BND portfolio
**Metrics:** Total return, Sharpe Ratio, max drawdown
**Validation:** Historical performance simulation

## Technical Stack
| Component | Technology |
|------|---------|
| **Data Collection** | yfinance API |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn, plotly |
| **Time Series Models** | statsmodels, pmdarima, tensorflow |
| **Portfolio Optimization** | PyPortfolioOpt, cvxpy |
| **Development** | Jupyter, VS Code, Git |
| **Testing** | pytest |