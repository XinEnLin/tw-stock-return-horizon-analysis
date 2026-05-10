"""
Step 6：VIF 共線性診斷
=========================================================
研究：探討長期與短期技術面指標與基本面指標對公司股價報酬率之影響

【流程】
1. 讀取清洗後的回歸用寬表（季版 + 年版）
2. 計算 11 個自變數的 VIF
3. 若有 VIF > 門檻，逐步剔除最高者，重新計算
4. 輸出最終保留的變數清單與 VIF 表

【公式】
VIF_i = 1 / (1 - R_i^2)
其中 R_i^2 是把 X_i 當應變數、對其他自變數做回歸所得到的判定係數。

【判讀標準】
VIF < 5：無共線性問題
VIF 5~10：中度共線（本研究採用之較嚴格門檻）
VIF > 10：嚴重共線

本研究採用 VIF > 5 之較嚴格標準，以確保模型估計品質。

【輸出】
- vif_quarterly.csv      季版最終 VIF 表
- vif_yearly.csv         年版最終 VIF 表
- vif_log.txt            完整剔除過程記錄
=========================================================

【套件需求】
pip install statsmodels
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from io import StringIO
import os

# =========================================================
# 設定區
# =========================================================
DATA_FOLDER   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_data")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vif_diagnosis")
FILE_Q = "regression_quarterly_clean.csv"
FILE_Y = "regression_yearly_clean.csv"

# 確保輸出資料夾存在
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 自變數清單（不含 RET 應變數）
X_VARS = ["ROA", "EPS", "ROE", "OM", "FCF",
          "BIAS", "PPO", "K", "RSI",
          "SIZE", "LEV"]

# VIF 剔除門檻（採用較嚴格之 5，確保無中度共線）
VIF_THRESHOLD = 5.0

# 顯示設定
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)


# =========================================================
# 核心函式
# =========================================================
def calc_vif(df, x_vars):
    """
    計算 VIF。注意要先加常數項（intercept），不然 VIF 算法不對。
    """
    X = sm.add_constant(df[x_vars])
    vif_data = pd.DataFrame({
        "Variable": x_vars,
        # i+1 是因為要跳過 const 那一欄
        "VIF": [variance_inflation_factor(X.values, i + 1)
                for i in range(len(x_vars))]
    })
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def severity(vif):
    if vif >= 10:
        return "[SEVERE]"
    elif vif >= 5:
        return "[MODERATE]"
    else:
        return "[OK]"


def iterative_removal(df, x_vars, threshold, log):
    """逐步剔除 VIF > threshold 的變數"""
    current = list(x_vars)
    iter_no = 0
    while True:
        iter_no += 1
        vif = calc_vif(df, current)
        max_vif = vif.iloc[0]["VIF"]

        log.write(f"\n--- 第 {iter_no} 輪 ---\n")
        log.write(vif.to_string(index=False) + "\n")

        if max_vif < threshold:
            log.write(f"\n>>> 全部 VIF < {threshold}，剔除完成\n")
            print(f"  第 {iter_no} 輪：全部 VIF < {threshold}，停止")
            return current, vif

        worst = vif.iloc[0]["Variable"]
        log.write(f">>> 最高 VIF：{worst} = {max_vif:.2f}（>{threshold}），剔除\n")
        print(f"  第 {iter_no} 輪：剔除 {worst}（VIF = {max_vif:.2f}）")
        current.remove(worst)


def diagnose(df, x_vars, threshold, label, log):
    """完整 VIF 診斷流程，回傳最終 VIF 表"""
    log.write("\n" + "=" * 80 + "\n")
    log.write(f"【{label}】VIF 共線性診斷\n")
    log.write("=" * 80 + "\n")

    print(f"\n{'='*60}")
    print(f"【{label}】")
    print(f"{'='*60}")

    # 初始 VIF
    log.write(f"\n初始 VIF（{len(x_vars)} 變數全在）：\n")
    init = calc_vif(df, x_vars)
    log.write(init.to_string(index=False) + "\n")
    print(f"\n初始 VIF（{len(x_vars)} 變數）：")
    print(init.to_string(index=False))

    # 逐步剔除
    print(f"\n逐步剔除（門檻 VIF > {threshold}）：")
    final_vars, final_vif = iterative_removal(df, x_vars, threshold, log)

    # 整理最終結果
    final_vif["Severity"] = final_vif["VIF"].apply(severity)
    log.write(f"\n--- 最終結果 ---\n")
    log.write(f"保留變數（{len(final_vars)} 個）：{final_vars}\n")
    removed = set(x_vars) - set(final_vars)
    log.write(f"剔除變數（{len(removed)} 個）：{removed if removed else '無'}\n")
    log.write(final_vif.to_string(index=False) + "\n")

    print(f"\n最終保留 {len(final_vars)} 個變數：")
    print(final_vif.to_string(index=False))
    print(f"剔除：{removed if removed else '無'}")

    return final_vars, final_vif


# =========================================================
# 主流程
# =========================================================
log = StringIO()
log.write(f"VIF 共線性診斷 (門檻：VIF > {VIF_THRESHOLD})\n")

# 讀資料
df_q = pd.read_csv(os.path.join(DATA_FOLDER, FILE_Q))
df_y = pd.read_csv(os.path.join(DATA_FOLDER, FILE_Y))

# 季版
final_q, vif_q = diagnose(df_q, X_VARS, VIF_THRESHOLD, "季版", log)

# 年版
final_y, vif_y = diagnose(df_y, X_VARS, VIF_THRESHOLD, "年版", log)


# =========================================================
# 輸出
# =========================================================
out_q   = os.path.join(OUTPUT_FOLDER, "vif_quarterly.csv")
out_y   = os.path.join(OUTPUT_FOLDER, "vif_yearly.csv")
out_log = os.path.join(OUTPUT_FOLDER, "vif_log.txt")

vif_q.to_csv(out_q, index=False, encoding="utf-8-sig")
vif_y.to_csv(out_y, index=False, encoding="utf-8-sig")

with open(out_log, "w", encoding="utf-8") as f:
    f.write(log.getvalue())

print(f"\n=== 完成 ===")
print(f"季版 VIF 表：{out_q}")
print(f"年版 VIF 表：{out_y}")
print(f"完整 log：{out_log}")
print(f"\n季版進入回歸的變數：{final_q}")
print(f"年版進入回歸的變數：{final_y}")

# 給使用者參考的下一步
print("\n" + "=" * 60)
print("論文寫作建議句")
print("=" * 60)
removed_q = set(X_VARS) - set(final_q)
removed_y = set(X_VARS) - set(final_y)
if not removed_q and not removed_y:
    print("「本研究採用 VIF > 5 之較嚴格門檻進行共線性診斷。經檢驗，")
    print(" 所有變數之 VIF 均低於 5，故保留全部 11 個自變數進入後續多元線性迴歸分析。」")
else:
    print(f"季版剔除：{removed_q if removed_q else '無'}")
    print(f"年版剔除：{removed_y if removed_y else '無'}")
    print()
    print("論文寫作參考（請依實際結果調整）：")
    print("「本研究採用較嚴格之 VIF > 5 門檻進行共線性診斷，以確保迴歸係數估計之穩健性。")
    print(f" 經逐步剔除程序，季版與年版模型分別剔除 {sorted(removed_q)} 與 {sorted(removed_y)}，")
    print(" 剔除後所有保留變數之 VIF 均小於 5。本決策同時呼應相關係數矩陣之發現——")
    print(" ROA 與 ROE 之 Pearson 相關係數高達 0.93，兩者作為獲利能力指標存在高度概念重疊，")
    print(" 故剔除 ROA 後，ROE 之 VIF 由原本接近 9 大幅下降至 2.2~2.3。」")