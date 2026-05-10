import pandas as pd
import glob
import os

# === 1. 讀取所有年度檔，合併成一張大表 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "data")
files = glob.glob(os.path.join(folder, "year*.csv"))
df_list = [pd.read_csv(f, encoding="utf-16", sep="\t") for f in files]
df = pd.concat(df_list, ignore_index=True)

# === 2. 統一欄位名稱 ===
df = df.rename(columns={
    "證券代碼": "coid",
    "年月日": "date",
    "收盤價(元)": "close"
})

df["coid"] = df["coid"].astype(str).str.split().str[0]
df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
df = df.sort_values(["coid", "date"]).reset_index(drop=True)

# === 3. 分組計算 EMA12、EMA26，以及 PPO（跨公司可比的標準化版本）===
# PPO = (EMA12 - EMA26) / EMA26 * 100，單位為%，消除股價水準差異
df["ema12"] = df.groupby("coid")["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
df["ema26"] = df.groupby("coid")["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df["ppo"] = (df["ema12"] - df["ema26"]) / df["ema26"] * 100

# === 4. 砍掉 warm-up 期間（保留 2020/01/01 之後）===
df = df[df["date"] >= "2020-01-01"].copy()

# === 5. 加上「年-季」欄位，取季末最後一個交易日的 PPO ===
# 季末值反映「進入下一季時的動能狀態」，與 BIAS、KD、RSI 慣例一致
df["year_q"] = df["date"].dt.to_period("Q").astype(str)   # 例如 2020Q1

quarterly = (
    df.sort_values(["coid", "date"])
      .groupby(["coid", "year_q"])
      .last()
      .reset_index()
      [["coid", "year_q", "date", "ppo"]]
      .rename(columns={"date": "quarter_end_date", "ppo": "ppo_quarter_end"})
)

# === 6. 年平均 PPO ===
df["year"] = df["date"].dt.year
yearly = (
    df.groupby(["coid", "year"])["ppo"]
      .mean()
      .reset_index()
      .rename(columns={"ppo": "ppo_year_avg"})
)

# === 7. 輸出成 CSV ===
quarterly_path = os.path.join(script_dir, "ppo_quarterly.csv")
yearly_path = os.path.join(script_dir, "ppo_yearly.csv")
quarterly.to_csv(quarterly_path, index=False, encoding="utf-8-sig")
yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
print("完成！季末 PPO：", quarterly_path)
print("完成！年平均 PPO：", yearly_path)
print(yearly.head(10))