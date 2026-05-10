import pandas as pd

# ---------- Quarterly ----------
ret_q = pd.read_csv("Stock_Return/quarterly.csv", dtype={"coid": str})
fund_q = pd.read_csv("Fundamentals/quarterly.csv", dtype={"coid": str})
bias_q = pd.read_csv("Technical/bias/bias_quarterly.csv", dtype={"coid": str})
ppo_q = pd.read_csv("Technical/ppo/ppo_quarterly.csv", dtype={"coid": str})
kd_q = pd.read_csv("Technical/KD/kd_quarterly.csv", dtype={"coid": str})
rsi_q = pd.read_csv("Technical/RSI/rsi_quarterly.csv", dtype={"coid": str})

bias_q = bias_q.rename(columns={"Bias": "BIAS"})
ppo_q = ppo_q.rename(columns={"ppo_quarter_end": "PPO"})
kd_q = kd_q.rename(columns={"K_avg": "K"})

key_q = ["coid", "year_q"]
df_q = (
    ret_q
    .merge(fund_q[key_q + ["ROA", "EPS", "ROE", "OM", "FCF", "total_assets", "LEV"]], on=key_q, how="left")
    .merge(bias_q, on=key_q, how="left")
    .merge(ppo_q, on=key_q, how="left")
    .merge(kd_q, on=key_q, how="left")
    .merge(rsi_q, on=key_q, how="left")
)
cols_q = key_q + ["RET", "ROA", "EPS", "ROE", "OM", "FCF", "total_assets", "LEV", "BIAS", "PPO", "K", "RSI"]
df_q = df_q[cols_q].sort_values(key_q).reset_index(drop=True)
df_q.to_csv("regression_quarterly.csv", index=False)
print(f"quarterly: {df_q.shape[0]} rows × {df_q.shape[1]} cols")
print(df_q.head(3).to_string())

# ---------- Yearly ----------
ret_y = pd.read_csv("Stock_Return/yearly.csv", dtype={"coid": str})
fund_y = pd.read_csv("Fundamentals/yearly.csv", dtype={"coid": str})
bias_y = pd.read_csv("Technical/bias/bias_yearly.csv", dtype={"coid": str})
ppo_y = pd.read_csv("Technical/ppo/ppo_yearly.csv", dtype={"coid": str})
kd_y = pd.read_csv("Technical/KD/kd_yearly.csv", dtype={"coid": str})
rsi_y = pd.read_csv("Technical/RSI/rsi_yearly.csv", dtype={"coid": str})

bias_y = bias_y.rename(columns={"Bias": "BIAS"})
ppo_y = ppo_y.rename(columns={"ppo_year_avg": "PPO"})
kd_y = kd_y.rename(columns={"K_avg": "K"})

key_y = ["coid", "year"]
df_y = (
    ret_y
    .merge(fund_y[key_y + ["ROA", "EPS", "ROE", "OM", "FCF", "total_assets", "LEV"]], on=key_y, how="left")
    .merge(bias_y, on=key_y, how="left")
    .merge(ppo_y, on=key_y, how="left")
    .merge(kd_y, on=key_y, how="left")
    .merge(rsi_y, on=key_y, how="left")
)
cols_y = key_y + ["RET", "ROA", "EPS", "ROE", "OM", "FCF", "total_assets", "LEV", "BIAS", "PPO", "K", "RSI"]
df_y = df_y[cols_y].sort_values(key_y).reset_index(drop=True)
df_y.to_csv("regression_yearly.csv", index=False)
print(f"\nyearly: {df_y.shape[0]} rows × {df_y.shape[1]} cols")
print(df_y.head(3).to_string())
