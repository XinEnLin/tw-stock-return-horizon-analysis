"""
Step 7：OLS 多元線性迴歸 + 預測誤差評估
=========================================================
研究：探討長期與短期技術面指標與基本面指標對公司股價報酬率之影響

【流程】
1. 讀取清洗後的回歸用寬表（季版 + 年版）
2. 對 RET 跑 OLS 多元線性迴歸（10 個自變數，已剔除 ROA）
   - (A) 原始迴歸：解釋「每單位自變數的實質影響」
   - (B) 標準化迴歸：所有變數轉 z-score，係數可直接比較「相對重要性」
3. 輸出回歸係數 β、SE、t 值、p 值、顯著性、R²、Adj R²、F 統計量
4. 計算預測誤差：MSE、MAD、MAPE（含 sMAPE 與過濾版）
5. 季版 vs 年版對照、檢驗論文假說 H1~H4
6. 分組模型比較：分別跑「只有基本面」「只有技術面」「完整模型」三版本，
   對照各自的 Adjusted R²，用於證明「基本面看長、技術面看短」

【為什麼要做標準化迴歸】
原始迴歸中，PPO 的數值範圍很小（約 ±0.05），導致其 β 高達 61，
無法與 RSI（0~100，β≈0.3）等變數直接比較大小。
標準化迴歸將所有變數轉為 z-score（平均 0、標準差 1），
此時係數（standardized beta）單位統一，可直接比較相對重要性。
標準化「不會」改變 R²、t 值、p 值與顯著性，只改變係數的尺度。
附帶好處：可大幅改善原始模型的 condition number 警告。

【為什麼要做分組模型比較】
完整模型的 Adjusted R² 受技術面指標（尤其 PPO）與報酬率之機械相關影響而偏高，
無法反映基本面的真實解釋力。將自變數分組後分別跑回歸，
可觀察「基本面解釋力」在季 / 年模型間的相對變化——
若基本面模型的 Adj R² 在年版明顯高於季版，即支持「基本面看長」之假說。

【迴歸式】
RET = β0 + β1*EPS + β2*ROE + β3*OM + β4*FCF
    + β5*BIAS + β6*PPO + β7*K + β8*RSI
    + β9*SIZE + β10*LEV + ε

【輸出】
- ols_quarterly.csv              季版原始迴歸係數表
- ols_yearly.csv                 年版原始迴歸係數表
- ols_quarterly_standardized.csv 季版標準化迴歸係數表
- ols_yearly_standardized.csv    年版標準化迴歸係數表
- ols_comparison.csv             季 / 年 β 對照表（含假說檢驗）
- ols_model_stats.csv            模型整體統計（R²、F、誤差指標）
- ols_grouped_comparison.csv     分組模型比較（基本面 / 技術面 / 完整）
- ols_log.txt                    完整 statsmodels summary（原始 + 標準化）
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
# 分三組，方便分組模型比較
FUND_VARS = ["EPS", "ROE", "OM", "FCF"]        # 基本面
TECH_VARS = ["BIAS", "PPO", "K", "RSI"]        # 技術面
CTRL_VARS = ["SIZE", "LEV"]                     # 控制
X_VARS = FUND_VARS + TECH_VARS + CTRL_VARS      # 完整 10 變數
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


def standardize(df, cols):
    """把指定欄位轉成 z-score（平均 0、標準差 1）"""
    df_std = df.copy()
    for col in cols:
        df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()
    return df_std


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


def fit_ols(df, x_vars, y_var):
    """跑一個 OLS，回傳 model 物件"""
    X = sm.add_constant(df[x_vars])
    y = df[y_var]
    return sm.OLS(y, X).fit()


def coef_table(model, x_vars):
    """把 model 的係數整理成 DataFrame"""
    coef_df = pd.DataFrame({
        "Variable": ["const"] + x_vars,
        "類別":     ["—"] + [VAR_CATEGORY.get(v, "—") for v in x_vars],
        "β":        model.params.values,
        "Std_Err":  model.bse.values,
        "t":        model.tvalues.values,
        "p":        model.pvalues.values,
    })
    coef_df["Sig"] = coef_df["p"].apply(sig_marker)
    return coef_df


def run_ols(df, x_vars, y_var, label, log):
    """
    執行 OLS 並輸出格式化報表。
    同時跑「原始迴歸」與「標準化迴歸」。
    """
    log.write("\n" + "=" * 90 + "\n")
    log.write(f"【{label}】OLS 回歸結果\n")
    log.write("=" * 90 + "\n")

    print(f"\n{'='*70}\n【{label}】\n{'='*70}")

    # ---------- (A) 原始迴歸 ----------
    model_raw = fit_ols(df, x_vars, y_var)
    coef_raw  = coef_table(model_raw, x_vars)

    log.write("\n--- (A) 原始迴歸 ---\n")
    log.write(str(model_raw.summary()) + "\n")

    print("\n>>> (A) 原始迴歸係數（解釋每單位實質影響）")
    print(coef_raw.round(4).to_string(index=False))

    # ---------- (B) 標準化迴歸 ----------
    # 把所有自變數 + 應變數都轉成 z-score 再跑
    df_std = standardize(df, x_vars + [y_var])
    model_std = fit_ols(df_std, x_vars, y_var)
    coef_std  = coef_table(model_std, x_vars)

    log.write("\n--- (B) 標準化迴歸 ---\n")
    log.write(str(model_std.summary()) + "\n")

    print("\n>>> (B) 標準化迴歸係數（比較相對重要性，可直接比大小）")
    # 標準化迴歸的 const 會趨近 0，且重點是 β 大小，所以只顯示關鍵欄位
    coef_std_show = coef_std[coef_std["Variable"] != "const"].copy()
    coef_std_show = coef_std_show.reindex(
        coef_std_show["β"].abs().sort_values(ascending=False).index
    )
    print(coef_std_show[["Variable", "類別", "β", "p", "Sig"]].round(4).to_string(index=False))
    print(f"   (依標準化 β 絕對值大小排序；標準化後 const ≈ 0，已略)")

    # ---------- 模型整體統計（原始與標準化的 R²/F/p 完全相同）----------
    print(f"\n>>> 模型整體（原始與標準化迴歸的 R²、F、p 值相同）")
    print(f"   N               : {int(model_raw.nobs):,}")
    print(f"   R²              : {model_raw.rsquared:.4f}")
    print(f"   Adjusted R²     : {model_raw.rsquared_adj:.4f}")
    print(f"   F-statistic     : {model_raw.fvalue:.2f}")
    print(f"   Prob (F)        : {model_raw.f_pvalue:.4e}")
    print(f"   Cond. No. 原始  : {model_raw.condition_number:.1f}")
    print(f"   Cond. No. 標準化: {model_std.condition_number:.1f}   ← 標準化後大幅改善")

    # ---------- 預測誤差（用原始迴歸計算，因為要還原成 RET 的實際單位）----------
    X_raw = sm.add_constant(df[x_vars])
    y_pred = model_raw.predict(X_raw)
    errors = calc_errors(df[y_var], y_pred)
    print(f"\n>>> 預測誤差（基於原始迴歸，單位為 RET 實際值）")
    print(f"   MSE                : {errors['MSE']:.4f}")
    print(f"   MAD                : {errors['MAD']:.4f}")
    print(f"   MAPE (原始)        : {errors['MAPE_raw']:.2f}%   ← 接近 0 的 RET 會放大")
    print(f"   MAPE (|y|>1% 過濾) : {errors['MAPE_filter']:.2f}%   ← 建議論文採用")
    print(f"   sMAPE (對稱)       : {errors['sMAPE']:.2f}%")

    # ---------- 整體統計打包 ----------
    stats = {
        "N":               int(model_raw.nobs),
        "R_squared":       model_raw.rsquared,
        "Adj_R_squared":   model_raw.rsquared_adj,
        "F_stat":          model_raw.fvalue,
        "F_pvalue":        model_raw.f_pvalue,
        "CondNo_raw":      model_raw.condition_number,
        "CondNo_std":      model_std.condition_number,
        **errors,
    }
    return model_raw, model_std, coef_raw, coef_std, stats


def run_grouped_comparison(df_q, df_y, y_var, log):
    """
    分組模型比較：分別跑三個版本的模型，對照 Adjusted R²。
    - 基本面 + 控制
    - 技術面 + 控制
    - 完整模型
    用於證明「基本面看長、技術面看短」。
    """
    log.write("\n" + "=" * 90 + "\n")
    log.write("分組模型比較（基本面 / 技術面 / 完整模型）\n")
    log.write("=" * 90 + "\n")

    print("\n" + "=" * 100)
    print("分組模型比較：基本面 vs 技術面 vs 完整模型")
    print("=" * 100)
    print("用途：完整模型的 R² 受 PPO 機械相關影響而偏高，分組後可看各類指標的真實解釋力\n")

    # 四個模型的變數組合
    # 「完整模型 (排除 PPO)」用於檢視 PPO（與 RET 有機械相關）對 R² 的影響：
    # 拿掉 PPO 後 Adj R² 下降幅度，即 PPO 單一變數的增量解釋力
    model_specs = {
        "基本面 + 控制":       FUND_VARS + CTRL_VARS,
        "技術面 + 控制":       TECH_VARS + CTRL_VARS,
        "完整模型":            FUND_VARS + TECH_VARS + CTRL_VARS,
        "完整模型 (排除 PPO)": FUND_VARS + [v for v in TECH_VARS if v != "PPO"] + CTRL_VARS,
    }

    rows = []
    for period_label, df in [("季版（短期）", df_q), ("年版（長期）", df_y)]:
        for model_name, x_vars in model_specs.items():
            m = fit_ols(df, x_vars, y_var)
            rows.append({
                "期間":        period_label,
                "模型":        model_name,
                "變數數":      len(x_vars),
                "R_squared":   m.rsquared,
                "Adj_R2":      m.rsquared_adj,
                "F_stat":      m.fvalue,
                "F_pvalue":    m.f_pvalue,
            })

    result_df = pd.DataFrame(rows)

    # 印出表格
    print(result_df.round(4).to_string(index=False))
    log.write(result_df.round(4).to_string(index=False) + "\n")

    # ---------- 關鍵對比：基本面解釋力的季 / 年變化 ----------
    print("\n--- 關鍵發現 ---")

    fund_q = result_df[(result_df["期間"]=="季版（短期）") &
                       (result_df["模型"]=="基本面 + 控制")]["Adj_R2"].values[0]
    fund_y = result_df[(result_df["期間"]=="年版（長期）") &
                       (result_df["模型"]=="基本面 + 控制")]["Adj_R2"].values[0]
    tech_q = result_df[(result_df["期間"]=="季版（短期）") &
                       (result_df["模型"]=="技術面 + 控制")]["Adj_R2"].values[0]
    tech_y = result_df[(result_df["期間"]=="年版（長期）") &
                       (result_df["模型"]=="技術面 + 控制")]["Adj_R2"].values[0]

    print(f"  基本面模型 Adj R²：季版 {fund_q:.4f} → 年版 {fund_y:.4f}", end="")
    if fund_q > 0:
        print(f"（提升 {fund_y/fund_q:.1f} 倍）")
    else:
        print()
    print(f"  技術面模型 Adj R²：季版 {tech_q:.4f} → 年版 {tech_y:.4f}"
          f"（變化 {(tech_y-tech_q):+.4f}）")

    print()
    if fund_y > fund_q:
        print("  → 基本面解釋力在長期（年）明顯高於短期（季），【支持「基本面看長」】")
    if abs(tech_y - tech_q) < 0.05:
        print("  → 技術面解釋力在長短期相近，且兩者皆高，【支持「技術面在各期間均有效」】")

    # ---------- PPO 增量解釋力：完整模型 vs 完整模型(排除 PPO) ----------
    full_q = result_df[(result_df["期間"]=="季版（短期）") &
                       (result_df["模型"]=="完整模型")]["Adj_R2"].values[0]
    full_y = result_df[(result_df["期間"]=="年版（長期）") &
                       (result_df["模型"]=="完整模型")]["Adj_R2"].values[0]
    noppo_q = result_df[(result_df["期間"]=="季版（短期）") &
                        (result_df["模型"]=="完整模型 (排除 PPO)")]["Adj_R2"].values[0]
    noppo_y = result_df[(result_df["期間"]=="年版（長期）") &
                        (result_df["模型"]=="完整模型 (排除 PPO)")]["Adj_R2"].values[0]

    print()
    print(f"  PPO 增量解釋力（完整模型 → 排除 PPO 後的 Adj R² 下降幅度）：")
    print(f"    季版：{full_q:.4f} → {noppo_q:.4f}（下降 {full_q - noppo_q:.4f}）")
    print(f"    年版：{full_y:.4f} → {noppo_y:.4f}（下降 {full_y - noppo_y:.4f}）")
    print(f"  → PPO 單一變數即貢獻完整模型約 {(full_q - noppo_q):.0%}~{(full_y - noppo_y):.0%} 的解釋力，")
    print(f"     此高貢獻部分肇因於 PPO 與當期 RET 之機械相關，論文宜於研究限制說明。")

    return result_df


# =========================================================
# 主流程
# =========================================================
log = StringIO()

df_q = pd.read_csv(os.path.join(DATA_FOLDER, FILE_Q))
df_y = pd.read_csv(os.path.join(DATA_FOLDER, FILE_Y))

print(f"季版資料：{len(df_q):,} 筆 | 年版資料：{len(df_y):,} 筆")

# 跑回歸（季版、年版，各自包含原始 + 標準化）
model_q_raw, model_q_std, coef_q_raw, coef_q_std, stats_q = \
    run_ols(df_q, X_VARS, Y_VAR, "季版（短期模型）", log)
model_y_raw, model_y_std, coef_y_raw, coef_y_std, stats_y = \
    run_ols(df_y, X_VARS, Y_VAR, "年版（長期模型）", log)


# =========================================================
# 季版 vs 年版對照表（檢驗 H1~H4）
# 同時列出「原始 β」與「標準化 β」
# =========================================================
print("\n" + "=" * 100)
print("【季版 vs 年版】β 對照（檢驗論文假說 H1~H4）")
print("=" * 100)

cmp = pd.DataFrame({
    "Variable":      X_VARS,
    "類別":          [VAR_CATEGORY[v] for v in X_VARS],
    "季_原始β":      [model_q_raw.params[v] for v in X_VARS],
    "季_標準化β":    [model_q_std.params[v] for v in X_VARS],
    "季_p":          [model_q_raw.pvalues[v] for v in X_VARS],
    "季_Sig":        [sig_marker(model_q_raw.pvalues[v]) for v in X_VARS],
    "年_原始β":      [model_y_raw.params[v] for v in X_VARS],
    "年_標準化β":    [model_y_std.params[v] for v in X_VARS],
    "年_p":          [model_y_raw.pvalues[v] for v in X_VARS],
    "年_Sig":        [sig_marker(model_y_raw.pvalues[v]) for v in X_VARS],
})
print(cmp.round(4).to_string(index=False))

# 各模型「相對重要性」排名（依標準化 β 絕對值）
print("\n--- 相對重要性排名（依標準化 β 絕對值；數值越大影響越大）---")
for label, model_std in [("季版", model_q_std), ("年版", model_y_std)]:
    ranking = pd.Series({v: abs(model_std.params[v]) for v in X_VARS})
    ranking = ranking.sort_values(ascending=False)
    rank_str = "  ".join([f"{i+1}.{v}({val:.3f})"
                          for i, (v, val) in enumerate(ranking.items())])
    print(f"  [{label}] {rank_str}")


# =========================================================
# 假說檢驗摘要
# =========================================================
print("\n" + "=" * 100)
print("假說檢驗摘要")
print("=" * 100)
print("(預測：β > 0 且顯著 → 支持假說；β < 0 或不顯著 → 不支持)")
print("(註：顯著性看 p 值；原始 β 與標準化 β 的正負號與 p 值完全一致)\n")


def hypothesis_check(varlist, model_raw, model_std, label):
    print(f"  {label}：")
    for v in varlist:
        b_raw = model_raw.params[v]
        b_std = model_std.params[v]
        p = model_raw.pvalues[v]
        sig = sig_marker(p)
        result = ("支持" if (b_raw > 0 and p < 0.05)
                  else "顯著但反向" if (b_raw < 0 and p < 0.05)
                  else "不顯著")
        print(f"    {v:5s}  原始β={b_raw:+.4f}  標準化β={b_std:+.4f}  "
              f"p={p:.4f}  {sig:3s}  → {result}")


print("\n[H2] 短期基本面 → RET 正相關（季版 EPS, ROE, OM, FCF）")
hypothesis_check(["EPS","ROE","OM","FCF"], model_q_raw, model_q_std, "季版基本面")

print("\n[H1] 長期基本面 → RET 正相關（年版 EPS, ROE, OM, FCF）")
hypothesis_check(["EPS","ROE","OM","FCF"], model_y_raw, model_y_std, "年版基本面")

print("\n[H4] 短期技術面 → RET 正相關（季版 BIAS, PPO, K, RSI）")
hypothesis_check(["BIAS","PPO","K","RSI"], model_q_raw, model_q_std, "季版技術面")

print("\n[H3] 長期技術面 → RET 正相關（年版 BIAS, PPO, K, RSI）")
hypothesis_check(["BIAS","PPO","K","RSI"], model_y_raw, model_y_std, "年版技術面")


# =========================================================
# 分組模型比較（基本面 / 技術面 / 完整模型）
# =========================================================
grouped_df = run_grouped_comparison(df_q, df_y, Y_VAR, log)


# =========================================================
# 輸出 CSV（給論文表格用）
# =========================================================
out_q       = os.path.join(OUTPUT_FOLDER, "ols_quarterly.csv")
out_y       = os.path.join(OUTPUT_FOLDER, "ols_yearly.csv")
out_q_std   = os.path.join(OUTPUT_FOLDER, "ols_quarterly_standardized.csv")
out_y_std   = os.path.join(OUTPUT_FOLDER, "ols_yearly_standardized.csv")
out_cmp     = os.path.join(OUTPUT_FOLDER, "ols_comparison.csv")
out_stats   = os.path.join(OUTPUT_FOLDER, "ols_model_stats.csv")
out_grouped = os.path.join(OUTPUT_FOLDER, "ols_grouped_comparison.csv")
out_log     = os.path.join(OUTPUT_FOLDER, "ols_log.txt")

# 原始迴歸係數表
coef_q_raw.round(4).to_csv(out_q, index=False, encoding="utf-8-sig")
coef_y_raw.round(4).to_csv(out_y, index=False, encoding="utf-8-sig")

# 標準化迴歸係數表
coef_q_std.round(4).to_csv(out_q_std, index=False, encoding="utf-8-sig")
coef_y_std.round(4).to_csv(out_y_std, index=False, encoding="utf-8-sig")

# 對照表
cmp.round(4).to_csv(out_cmp, index=False, encoding="utf-8-sig")

# 模型統計打包成一張表
stats_df = pd.DataFrame({
    "指標":   list(stats_q.keys()),
    "季版":   list(stats_q.values()),
    "年版":   list(stats_y.values()),
})
stats_df.round(4).to_csv(out_stats, index=False, encoding="utf-8-sig")

# 分組模型比較表
grouped_df.round(4).to_csv(out_grouped, index=False, encoding="utf-8-sig")

with open(out_log, "w", encoding="utf-8") as f:
    f.write(log.getvalue())

print(f"\n=== 完成 ===")
print(f"季版原始迴歸表   : {out_q}")
print(f"年版原始迴歸表   : {out_y}")
print(f"季版標準化迴歸表 : {out_q_std}")
print(f"年版標準化迴歸表 : {out_y_std}")
print(f"對照表           : {out_cmp}")
print(f"模型統計         : {out_stats}")
print(f"分組模型比較     : {out_grouped}")
print(f"完整 log         : {out_log}")