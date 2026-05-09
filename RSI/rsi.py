"""
Wilder's RSI 計算程式
=========================================================
研究：探討長期與短期技術面指標與基本面指標對公司股價報酬率之影響
資料來源：TEJ 台灣經濟新報資料庫

【資料需求】
- 範圍：2019/07/01 ~ 2025/12/31（含 warm-up 期間）
- 樣本：台灣上市、上櫃公司
- 欄位：證券代碼、年月日、收盤價(元)
- 檔案格式：CSV，依年度分檔放在同一資料夾

【輸出】
- rsi_quarterly.csv：欄位 coid、year_q、RSI（季末 Wilder's RSI 值）
- rsi_yearly.csv  ：欄位 coid、year、RSI（年末 Wilder's RSI 值）

【計算方法】
1. Gain  = max(今日收盤 − 昨日收盤, 0)
2. Loss  = max(昨日收盤 − 今日收盤, 0)
3. 採 Wilder's 平滑（α = 1/N 的 EMA），N = 14
4. RSI   = 100 × 平均漲幅 / (平均漲幅 + 平均跌幅)
5. 季末／年末值（每季／每年最後一個交易日的 RSI）
=========================================================
"""

import pandas as pd
import glob
import os

# =========================================================
# 設定區（請依實際路徑調整）
# =========================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER    = os.path.join(script_dir, "data")
OUTPUT_QUARTERLY = "rsi_quarterly.csv"
OUTPUT_YEARLY    = "rsi_yearly.csv"
N = 14                    # RSI 期數（業界慣例 14 日）
START_DATE = "2020-01-01" # 分析起始日（warm-up 期間之後）

# =========================================================
# 1. 讀取所有年度檔，合併成一張大表
# =========================================================
files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
print(f"讀取 {len(files)} 個檔案...")

df_list = [pd.read_csv(f, encoding="utf-16", sep="\t") for f in files]
df = pd.concat(df_list, ignore_index=True)

df = df.rename(columns={
    "證券代碼":   "coid",
    "年月日":    "date",
    "收盤價(元)": "close",
})

df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
df = df.sort_values(["coid", "date"]).reset_index(drop=True)
print(f"總筆數：{len(df):,} | 公司數：{df['coid'].nunique()}")

# =========================================================
# 2. 計算單日漲幅 (Gain) 與跌幅 (Loss)
# =========================================================
df["change"] = df.groupby("coid")["close"].diff()
df["gain"]   = df["change"].clip(lower=0)
df["loss"]   = (-df["change"]).clip(lower=0)

# =========================================================
# 3. Wilder's 平滑（α = 1/N 的指數平滑）
# =========================================================
df["avg_gain"] = df.groupby("coid")["gain"].transform(
    lambda x: x.ewm(alpha=1/N, adjust=False).mean()
)
df["avg_loss"] = df.groupby("coid")["loss"].transform(
    lambda x: x.ewm(alpha=1/N, adjust=False).mean()
)

# =========================================================
# 4. 計算 RSI
# =========================================================
denom = df["avg_gain"] + df["avg_loss"]
df["RSI"] = (df["avg_gain"] / denom * 100).where(denom != 0, 50)

# =========================================================
# 5. 砍 warm-up 期間
# =========================================================
df = df[df["date"] >= START_DATE].copy()

# =========================================================
# 6. 季末 RSI
# =========================================================
df["year_q"] = df["date"].dt.to_period("Q").astype(str)

quarterly = (
    df.sort_values(["coid", "year_q", "date"])
      .groupby(["coid", "year_q"])
      .tail(1)[["coid", "year_q", "RSI"]]
      .reset_index(drop=True)
)

# =========================================================
# 7. 年末 RSI
# =========================================================
df["year"] = df["date"].dt.year

yearly = (
    df.sort_values(["coid", "year", "date"])
      .groupby(["coid", "year"])
      .tail(1)[["coid", "year", "RSI"]]
      .reset_index(drop=True)
)

# =========================================================
# 8. 輸出
# =========================================================
quarterly_path = os.path.join(script_dir, OUTPUT_QUARTERLY)
yearly_path    = os.path.join(script_dir, OUTPUT_YEARLY)

quarterly.to_csv(quarterly_path, index=False, encoding="utf-8-sig")
yearly.to_csv(yearly_path,    index=False, encoding="utf-8-sig")

print(f"\n完成！")
print(f"  季 RSI → {quarterly_path}（{len(quarterly):,} 筆）")
print(f"  年 RSI → {yearly_path}（{len(yearly):,} 筆）")
print("\n季 RSI 前 5 筆：")
print(quarterly.head(5))
print("\n年 RSI 前 5 筆：")
print(yearly.head(5))