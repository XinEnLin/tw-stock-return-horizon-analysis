import pandas as pd
import glob, os

# === 1. 讀取所有年度檔（UTF-16 LE、Tab 分隔）===
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
files = glob.glob(os.path.join(folder, "*.csv"))
df = pd.concat(
    [pd.read_csv(f, encoding="utf-16", sep="\t") for f in files],
    ignore_index=True
)

df.columns = df.columns.str.strip()
df = df.rename(columns={
    "證券代碼": "coid",
    "年月日":   "date",
    "收盤價(元)": "close",
    "最高價(元)": "high",
    "最低價(元)": "low",
})

# 只保留股票代碼（去除後面的公司名稱）
df["coid"] = df["coid"].astype(str).str.strip().str.split().str[0]
df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
df = df.sort_values(["coid", "date"]).reset_index(drop=True)

# === 2. 9 日最高、最低（每家公司分開）===
df["high_9"] = df.groupby("coid")["high"].transform(lambda x: x.rolling(9, min_periods=9).max())
df["low_9"]  = df.groupby("coid")["low"].transform(lambda x: x.rolling(9, min_periods=9).min())

# === 3. RSV（處理分母為 0 的邊界）===
denom = df["high_9"] - df["low_9"]
df["rsv"] = ((df["close"] - df["low_9"]) / denom * 100).where(denom != 0, 50)

# === 4. K 與 D（α = 1/3 的 EMA，跨公司分組）===
df["K"] = df.groupby("coid")["rsv"].transform(lambda x: x.ewm(alpha=1/3, adjust=False).mean())

# === 5. 過濾 2020–2025 ===
df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2025-12-31")].copy()
df["year"]   = df["date"].dt.year
df["year_q"] = df["date"].dt.to_period("Q").astype(str)

# === 6. 季平均 K 值（每家公司、每季所有交易日的平均）===
quarterly_avg = (
    df.groupby(["coid", "year_q"])["K"]
      .mean()
      .rename("K_avg")
      .reset_index()
)

# === 7. 年平均 K 值（每家公司、每年所有交易日的平均）===
annual_avg = (
    df.groupby(["coid", "year"])["K"]
      .mean()
      .rename("K_avg")
      .reset_index()
)

# === 8. 輸出結果 ===
out_dir = os.path.dirname(os.path.abspath(__file__))
quarterly_path = os.path.join(out_dir, "kd_quarterly.csv")
annual_path    = os.path.join(out_dir, "kd_yearly.csv")

quarterly_avg.to_csv(quarterly_path, index=False, encoding="utf-8-sig")
annual_avg.to_csv(annual_path,    index=False, encoding="utf-8-sig")

print("=== 季平均K值（前10筆）===")
print(quarterly_avg.head(10).to_string(index=False))
print("\n=== 年平均K值（前10筆）===")
print(annual_avg.head(10).to_string(index=False))
print(f"\n季平均K值 -> {quarterly_path}")
print(f"年平均K值 -> {annual_path}")