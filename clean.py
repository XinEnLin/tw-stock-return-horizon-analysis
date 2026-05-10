import pandas as pd
import numpy as np

# df = pd.read_csv("regression_quarterly.csv").drop(columns=["total_assets"])
df = pd.read_csv("regression_yearly.csv").drop(columns=["total_assets"])

# 整理欄位順序
# cols = ["coid","year_q","RET","ROA","EPS","ROE","OM","FCF",
#         "BIAS","PPO","K","RSI","SIZE","LEV"]
cols = ["coid","year","RET","ROA","EPS","ROE","OM","FCF",
        "BIAS","PPO","K","RSI","SIZE","LEV"]
df = df[cols]

# Step 3a: drop NaN
df = df.dropna()

# Step 3b: Winsorize 1% / 99%
VARS = ["RET","ROA","EPS","ROE","OM","FCF","BIAS","PPO","K","RSI","SIZE","LEV"]
for col in VARS:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lo, upper=hi)

# df.to_csv("regression_quarterly_clean.csv", index=False, encoding="utf-8-sig")
df.to_csv("regression_yearly_clean.csv", index=False, encoding="utf-8-sig")