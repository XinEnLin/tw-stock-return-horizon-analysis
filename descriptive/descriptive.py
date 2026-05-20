"""
Step 4：敘述統計
=========================================================
研究：探討長期與短期技術面指標與基本面指標對公司股價報酬率之影響

【流程】
1. 讀取清洗後的回歸用寬表（季版 + 年版）
2. 計算每個變數的敘述統計量：
   N、平均、標準差、最小、Q1、中位數、Q3、最大、偏態、峰態
3. 輸出成 CSV（給論文「表 1：敘述統計」用）

【統計量說明】
- count : 樣本數
- mean  : 平均數
- std   : 標準差
- min   : 最小值
- 25%   : 第一四分位數 (Q1)
- 50%   : 中位數 (Q2)
- 75%   : 第三四分位數 (Q3)
- max   : 最大值
- skew  : 偏態（> 0 右偏 / < 0 左偏；絕對值越大越不對稱）
- kurt  : 峰態（> 0 比常態尖峰厚尾；statsmodels/pandas 採超額峰態，常態 = 0）

【輸出】
- descriptive_quarterly.csv   季版敘述統計
- descriptive_yearly.csv      年版敘述統計
- descriptive_combined.csv    季 / 年合併對照（給論文表格用）
- descriptive_log.txt         統計結果資料（三張表的文字版）
=========================================================
"""

import pandas as pd
import numpy as np
import os

# =========================================================
# 設定區
# =========================================================
DATA_FOLDER   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clean_data")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "descriptive_stats")
FILE_Q = "regression_quarterly_clean.csv"
FILE_Y = "regression_yearly_clean.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 要做敘述統計的變數（含應變數 RET）
# 順序：應變數、基本面、技術面、控制
VARS = ["RET",                         # 應變數
        "EPS", "ROE", "OM", "FCF",      # 基本面
        "BIAS", "PPO", "K", "RSI",      # 技術面
        "SIZE", "LEV"]                   # 控制

VAR_CATEGORY = {
    "RET":"應變數",
    "EPS":"基本面", "ROE":"基本面", "OM":"基本面", "FCF":"基本面",
    "BIAS":"技術面","PPO":"技術面","K":"技術面",  "RSI":"技術面",
    "SIZE":"控制",  "LEV":"控制",
}

# 顯示設定
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


# =========================================================
# 核心函式
# =========================================================
def describe_table(df, varlist):
    """
    產生一張完整的敘述統計表。
    包含 pandas describe 的基本量 + 偏態 + 峰態。
    """
    # pandas describe 提供 count/mean/std/min/25%/50%/75%/max
    desc = df[varlist].describe(percentiles=[.25, .5, .75]).T

    # 額外補上偏態與峰態
    desc["skew"] = df[varlist].skew()
    desc["kurt"] = df[varlist].kurt()   # pandas 用超額峰態（常態 = 0）

    # 調整欄位順序
    desc = desc[["count", "mean", "std", "min",
                 "25%", "50%", "75%", "max", "skew", "kurt"]]

    # 補上變數類別欄位，放最前面
    desc.insert(0, "類別", [VAR_CATEGORY.get(v, "—") for v in desc.index])

    # 把 count 轉成整數比較好看
    desc["count"] = desc["count"].astype(int)

    return desc


# =========================================================
# 主流程
# =========================================================
print(">>> 讀取資料")
df_q = pd.read_csv(os.path.join(DATA_FOLDER, FILE_Q))
df_y = pd.read_csv(os.path.join(DATA_FOLDER, FILE_Y))
print(f"   季版：{len(df_q):,} 筆 | 年版：{len(df_y):,} 筆")

# 季版敘述統計
print("\n" + "=" * 95)
print("【季版（短期）】敘述統計")
print("=" * 95)
desc_q = describe_table(df_q, VARS)
print(desc_q.round(4).to_string())

# 年版敘述統計
print("\n" + "=" * 95)
print("【年版（長期）】敘述統計")
print("=" * 95)
desc_y = describe_table(df_y, VARS)
print(desc_y.round(4).to_string())


# =========================================================
# 合併對照表（季 / 年並排，給論文表格用）
# =========================================================
# 只挑論文表格常用的幾個欄位：N、mean、std、min、median、max
key_cols = ["count", "mean", "std", "min", "50%", "max"]

combined = pd.DataFrame({"類別": [VAR_CATEGORY.get(v, "—") for v in VARS]},
                        index=VARS)
for c in key_cols:
    combined[f"季_{c}"] = desc_q[c]
for c in key_cols:
    combined[f"年_{c}"] = desc_y[c]

print("\n" + "=" * 95)
print("【季 / 年對照】敘述統計（論文表格用，僅列關鍵欄位）")
print("=" * 95)
print(combined.round(4).to_string())


# =========================================================
# 輸出 CSV
# =========================================================
out_q   = os.path.join(OUTPUT_FOLDER, "descriptive_quarterly.csv")
out_y   = os.path.join(OUTPUT_FOLDER, "descriptive_yearly.csv")
out_cmb = os.path.join(OUTPUT_FOLDER, "descriptive_combined.csv")

# index 帶變數名稱，需要 index=True
desc_q.round(4).to_csv(out_q, encoding="utf-8-sig", index_label="Variable")
desc_y.round(4).to_csv(out_y, encoding="utf-8-sig", index_label="Variable")
combined.round(4).to_csv(out_cmb, encoding="utf-8-sig", index_label="Variable")

print(f"\n=== 完成 ===")
print(f"季版敘述統計   : {out_q}")
print(f"年版敘述統計   : {out_y}")
print(f"季/年對照表    : {out_cmb}")

# =========================================================
# 觀察提示
# =========================================================
print("\n" + "=" * 95)
print("判讀提示")
print("=" * 95)
print("""
1. 偏態 (skew)：絕對值 > 1 表示分布明顯不對稱。
   - 本資料 EPS、FCF、OM 偏態較大，反映財務指標常見的長尾特性。
   - 已做過 winsorize，若偏態仍大屬正常，OLS 對中度偏態具相當穩健性。

2. 峰態 (kurt)：> 0 表示比常態分布更尖峰厚尾（極端值較多）。
   - 若 RET 峰態偏高，可在論文說明並佐以大樣本中央極限定理支持 OLS 推論。

3. 平均 vs 中位數：兩者差距大代表分布偏斜，
   論文描述變數時建議兩者並陳。

4. 標準差：可看出各變數的離散程度差異極大
   （如 PPO 標準差約 2、SIZE 約 1.4、FCF 達 80+），
   這也再次說明跨變數比較係數時須採用標準化迴歸。
""")


# =========================================================
# 輸出 descriptive_log.txt（僅統計結果資料）
# =========================================================
out_log = os.path.join(OUTPUT_FOLDER, "descriptive_log.txt")

SEP = "=" * 95
lines = []

lines.append(SEP)
lines.append("【季版（短期）】敘述統計")
lines.append(SEP)
lines.append(desc_q.round(4).to_string())

lines.append("")
lines.append(SEP)
lines.append("【年版（長期）】敘述統計")
lines.append(SEP)
lines.append(desc_y.round(4).to_string())

lines.append("")
lines.append(SEP)
lines.append("【季 / 年對照】敘述統計（論文表格用，僅列關鍵欄位）")
lines.append(SEP)
lines.append(combined.round(4).to_string())

with open(out_log, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"統計結果 log   : {out_log}")