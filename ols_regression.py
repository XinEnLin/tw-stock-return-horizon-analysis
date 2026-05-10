"""
Step 7：OLS 多元線性迴歸 + 預測誤差評估
=========================================================
研究：探討長期與短期技術面指標與基本面指標對公司股價報酬率之影響

【流程】
1. 讀取清洗後的回歸用寬表（季版 + 年版）
2. 對 RET 跑 OLS 多元線性迴歸（10 個自變數，已剔除 ROA）
3. 輸出回歸係數 β、SE、t 值、p 值、顯著性、R²、Adj R²、F 統計量
4. 計算預測誤差：MSE、MAD、MAPE（含 sMAPE 與過濾版）
5. 季版 vs 年版對照、檢驗論文假說 H1~H4

【迴歸式】
RET = β0 + β1*EPS + β2*ROE + β3*OM + β4*FCF
    + β5*BIAS + β6*PPO + β7*K + β8*RSI
    + β9*SIZE + β10*LEV + ε

【輸出】
- ols_quarterly.csv          季版回歸係數表
- ols_yearly.csv             年版回歸係數表
- ols_comparison.csv         季 / 年 β 對照表（含假說檢驗）
- ols_model_stats.csv        模型整體統計（R²、F、誤差指標）
- ols_log.txt                完整 statsmodels summary
=========================================================

【套件需求】
pip install statsmodels
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from io import StringIO
import os

# =========================================================
# 設定區
# =========================================================
DATA_FOLDER   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_data")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ols_results")
FILE_Q = "regression_quarterly_clean.csv"
FILE_Y = "regression_yearly_clean.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# VIF > 5 篩選後的 10 個自變數（已剔除 ROA）
# 順序：基本面 4 個、技術面 4 個、控制 2 個
X_VARS = ["EPS", "ROE", "OM", "FCF",          # 基本面
          "BIAS", "PPO", "K", "RSI",          # 技術面
          "SIZE", "LEV"]                       # 控制
Y_VAR  = "RET"

# 變數類別對照（給輸出表用）
VAR_CATEGORY = {
    "EPS":"基本面", "ROE":"基本面", "OM":"基本面", "FCF":"基本面",
    "BIAS":"技術面","PPO":"技術面","K":"技術面",  "RSI":"技術面",
    "SIZE":"控制",  "LEV":"控制",
}

# 顯示設定
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


# =========================================================
# 工具函式
# =========================================================
def sig_marker(p):
    """產生顯著性星號"""
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def calc_errors(y_true, y_pred):
    """計算 MSE / MAD / MAPE / sMAPE / 過濾版 MAPE"""
    resid = y_true - y_pred
    mse = (resid**2).mean()
    mad = resid.abs().mean()

    # 標準 MAPE（會被接近 0 的 y 值炸開）
    mask_nonzero = y_true.abs() > 1e-6
    mape_raw = (resid[mask_nonzero].abs() / y_true[mask_nonzero].abs()).mean() * 100

    # 過濾版 MAPE：只算 |y| > 1% 的樣本（建議論文採用此版）
    mask_filter = y_true.abs() > 1.0
    if mask_filter.sum() > 0:
        mape_filter = (resid[mask_filter].abs() / y_true[mask_filter].abs()).mean() * 100
    else:
        mape_filter = np.nan

    # sMAPE（對稱 MAPE，分母用 (|y|+|ŷ|)/2，避免炸開）
    denom = (y_true.abs() + y_pred.abs()) / 2
    mask_smape = denom > 1e-6
    smape = (resid[mask_smape].abs() / denom[mask_smape]).mean() * 100

    return {
        "MSE":   mse,
        "MAD":   mad,
        "MAPE_raw":    mape_raw,
        "MAPE_filter": mape_filter,
        "sMAPE":       smape,
    }


def run_ols(df, x_vars, y_var, label, log):
    """執行 OLS 並輸出格式化報表"""
    log.write("\n" + "=" * 90 + "\n")
    log.write(f"【{label}】OLS 回歸結果\n")
    log.write("=" * 90 + "\n")

    print(f"\n{'='*70}\n【{label}】\n{'='*70}")

    X = sm.add_constant(df[x_vars])
    y = df[y_var]
    model = sm.OLS(y, X).fit()

    # 完整 statsmodels summary 寫入 log（可供論文附錄）
    log.write(str(model.summary()) + "\n")

    # 整理回歸係數表
    coef_df = pd.DataFrame({
        "Variable": ["const"] + x_vars,
        "類別":     ["—"] + [VAR_CATEGORY.get(v, "—") for v in x_vars],
        "β":        model.params.values,
        "Std_Err":  model.bse.values,
        "t":        model.tvalues.values,
        "p":        model.pvalues.values,
    })
    coef_df["Sig"] = coef_df["p"].apply(sig_marker)

    # 印出回歸係數表
    print("\n>>> 回歸係數")
    print(coef_df.round(4).to_string(index=False))

    # 模型整體統計
    print(f"\n>>> 模型整體")
    print(f"   N             : {int(model.nobs):,}")
    print(f"   R²            : {model.rsquared:.4f}")
    print(f"   Adjusted R²   : {model.rsquared_adj:.4f}")
    print(f"   F-statistic   : {model.fvalue:.2f}")
    print(f"   Prob (F)      : {model.f_pvalue:.4e}")

    # 預測誤差
    y_pred = model.predict(X)
    errors = calc_errors(y, y_pred)
    print(f"\n>>> 預測誤差")
    print(f"   MSE                : {errors['MSE']:.4f}")
    print(f"   MAD                : {errors['MAD']:.4f}")
    print(f"   MAPE (原始)        : {errors['MAPE_raw']:.2f}%   ← 接近 0 的 RET 會放大")
    print(f"   MAPE (|y|>1% 過濾) : {errors['MAPE_filter']:.2f}%   ← 建議論文採用")
    print(f"   sMAPE (對稱)       : {errors['sMAPE']:.2f}%")

    # 整體統計打包
    stats = {
        "N":             int(model.nobs),
        "R_squared":     model.rsquared,
        "Adj_R_squared": model.rsquared_adj,
        "F_stat":        model.fvalue,
        "F_pvalue":      model.f_pvalue,
        **errors,
    }
    return model, coef_df, stats


# =========================================================
# 主流程
# =========================================================
log = StringIO()

df_q = pd.read_csv(os.path.join(DATA_FOLDER, FILE_Q))
df_y = pd.read_csv(os.path.join(DATA_FOLDER, FILE_Y))

print(f"季版資料：{len(df_q):,} 筆 | 年版資料：{len(df_y):,} 筆\n")

# 跑回歸
model_q, coef_q, stats_q = run_ols(df_q, X_VARS, Y_VAR, "季版（短期模型）", log)
model_y, coef_y, stats_y = run_ols(df_y, X_VARS, Y_VAR, "年版（長期模型）", log)


# =========================================================
# 季版 vs 年版對照表（檢驗 H1~H4）
# =========================================================
print("\n" + "=" * 90)
print("【季版 vs 年版】β 對照（檢驗論文假說 H1~H4）")
print("=" * 90)

cmp = pd.DataFrame({
    "Variable": X_VARS,
    "類別":     [VAR_CATEGORY[v] for v in X_VARS],
    "季版_β":   [model_q.params[v] for v in X_VARS],
    "季版_p":   [model_q.pvalues[v] for v in X_VARS],
    "季版_Sig": [sig_marker(model_q.pvalues[v]) for v in X_VARS],
    "年版_β":   [model_y.params[v] for v in X_VARS],
    "年版_p":   [model_y.pvalues[v] for v in X_VARS],
    "年版_Sig": [sig_marker(model_y.pvalues[v]) for v in X_VARS],
})
print(cmp.round(4).to_string(index=False))


# =========================================================
# 假說檢驗摘要
# =========================================================
print("\n" + "=" * 90)
print("假說檢驗摘要")
print("=" * 90)
print("(預測：β > 0 且顯著 → 支持假說；β < 0 或不顯著 → 不支持)\n")


def hypothesis_check(varlist, model, label):
    print(f"  {label}：")
    for v in varlist:
        b = model.params[v]
        p = model.pvalues[v]
        sig = sig_marker(p)
        result = ("支持" if (b > 0 and p < 0.05)
                  else "顯著但反向" if (b < 0 and p < 0.05)
                  else "不顯著")
        print(f"    {v:5s}  β={b:+.4f}  p={p:.4f}  {sig:3s}  → {result}")


print("\n[H2] 短期基本面 → RET 正相關（季版 EPS, ROE, OM, FCF）")
hypothesis_check(["EPS","ROE","OM","FCF"], model_q, "季版基本面")

print("\n[H1] 長期基本面 → RET 正相關（年版 EPS, ROE, OM, FCF）")
hypothesis_check(["EPS","ROE","OM","FCF"], model_y, "年版基本面")

print("\n[H4] 短期技術面 → RET 正相關（季版 BIAS, PPO, K, RSI）")
hypothesis_check(["BIAS","PPO","K","RSI"], model_q, "季版技術面")

print("\n[H3] 長期技術面 → RET 正相關（年版 BIAS, PPO, K, RSI）")
hypothesis_check(["BIAS","PPO","K","RSI"], model_y, "年版技術面")


# =========================================================
# 輸出 CSV（給論文表格用）
# =========================================================
out_q     = os.path.join(OUTPUT_FOLDER, "ols_quarterly.csv")
out_y     = os.path.join(OUTPUT_FOLDER, "ols_yearly.csv")
out_cmp   = os.path.join(OUTPUT_FOLDER, "ols_comparison.csv")
out_stats = os.path.join(OUTPUT_FOLDER, "ols_model_stats.csv")
out_log   = os.path.join(OUTPUT_FOLDER, "ols_log.txt")

coef_q.round(4).to_csv(out_q, index=False, encoding="utf-8-sig")
coef_y.round(4).to_csv(out_y, index=False, encoding="utf-8-sig")
cmp.round(4).to_csv(out_cmp, index=False, encoding="utf-8-sig")

# 模型統計打包成一張表
stats_df = pd.DataFrame({
    "指標":   list(stats_q.keys()),
    "季版":   list(stats_q.values()),
    "年版":   list(stats_y.values()),
})
stats_df.round(4).to_csv(out_stats, index=False, encoding="utf-8-sig")

with open(out_log, "w", encoding="utf-8") as f:
    f.write(log.getvalue())

print(f"\n=== 完成 ===")
print(f"季版回歸表 : {out_q}")
print(f"年版回歸表 : {out_y}")
print(f"對照表     : {out_cmp}")
print(f"模型統計   : {out_stats}")
print(f"完整 log   : {out_log}")