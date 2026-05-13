"""
Step 5：相關係數矩陣分析與視覺化
=========================================================
研究：探討長期與短期技術面指標與基本面指標對公司股價報酬率之影響

【流程】
1. 讀取清洗後的回歸用寬表（季版 + 年版）
2. 計算 Pearson 相關係數矩陣
3. 找出高相關配對（共線性警示）
4. 列出 RET 與各自變數的相關度
5. 視覺化：熱力圖（季版 + 年版並排）
6. 輸出：相關矩陣 CSV + 熱力圖 PNG

【輸出檔案】
- correlation_quarterly.csv     季版相關矩陣
- correlation_yearly.csv        年版相關矩陣
- correlation_heatmap.png       並排熱力圖（給論文用）
=========================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")              # 不開圖形視窗，直接存檔
import matplotlib.pyplot as plt
import os

# =========================================================
# 設定區（請依實際路徑調整）
# =========================================================
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_data")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correlation_analysis")
FILE_Q = "regression_quarterly_clean.csv"
FILE_Y = "regression_yearly_clean.csv"

# 要納入分析的變數（順序會影響熱力圖排版）
ALL_VARS = ["RET", "ROA", "EPS", "ROE", "OM", "FCF",
            "BIAS", "PPO", "K", "RSI", "SIZE", "LEV"]

# 共線性警示門檻
WARN_THRESHOLD = 0.5    # 列出 |corr| >= 0.5 的配對
SEVERE_THRESHOLD = 0.8  # 標紅
MODERATE_THRESHOLD = 0.7

# 顯示設定
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


# =========================================================
# 1. 讀取資料
# =========================================================
def load(folder, fname):
    df = pd.read_csv(os.path.join(folder, fname))
    return df

print(">>> Step 1：讀取資料")
df_q = load(DATA_FOLDER, FILE_Q)
df_y = load(DATA_FOLDER, FILE_Y)
print(f"   季版：{len(df_q):,} 筆")
print(f"   年版：{len(df_y):,} 筆")


# =========================================================
# 2. 計算 Pearson 相關係數矩陣
# =========================================================
print("\n>>> Step 2：計算相關係數矩陣")
corr_q = df_q[ALL_VARS].corr()
corr_y = df_y[ALL_VARS].corr()

print("\n--- 【季版】 ---")
print(corr_q.round(3))
print("\n--- 【年版】 ---")
print(corr_y.round(3))


# =========================================================
# 3. 找出自變數間高相關配對（共線性警示）
# =========================================================
def find_high_corr(corr, threshold, exclude_col="RET"):
    """找出所有 |corr| >= threshold 的變數配對（排除應變數 RET）"""
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[i] == exclude_col or cols[j] == exclude_col:
                continue
            v = corr.iloc[i, j]
            if abs(v) >= threshold:
                pairs.append((cols[i], cols[j], v))
    return sorted(pairs, key=lambda x: -abs(x[2]))


def severity_label(v):
    a = abs(v)
    if a >= SEVERE_THRESHOLD:
        return "[SEVERE]"
    elif a >= MODERATE_THRESHOLD:
        return "[MODERATE]"
    else:
        return "[NOTE]"


print("\n>>> Step 3：自變數間 |相關係數| ≥ {:.1f} 的配對".format(WARN_THRESHOLD))
print("   (RET 為應變數，已排除)")
for name, corr in [("季版", corr_q), ("年版", corr_y)]:
    print(f"\n[{name}]")
    pairs = find_high_corr(corr, WARN_THRESHOLD)
    if pairs:
        for v1, v2, c in pairs:
            print(f"  {severity_label(c):11s}  {v1:5s} <-> {v2:5s}: {c:+.3f}")
    else:
        print("  (無)")


# =========================================================
# 4. RET 與各自變數的相關度
# =========================================================
print("\n>>> Step 4：RET 與各自變數的相關係數（依絕對值排序）")
for name, corr in [("季版", corr_q), ("年版", corr_y)]:
    print(f"\n[{name}]")
    ret_corr = corr["RET"].drop("RET").sort_values(key=abs, ascending=False)
    for v, c in ret_corr.items():
        bar = "█" * int(abs(c) * 30)         # 視覺化長條
        sign = "+" if c >= 0 else "-"
        print(f"  {v:6s}: {sign}{abs(c):.3f}  {bar}")


# =========================================================
# 5. 視覺化：並排熱力圖
# =========================================================
print("\n>>> Step 5：產生熱力圖")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, (title, corr) in zip(axes, [("Quarterly", corr_q), ("Yearly", corr_y)]):
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # 軸標籤
    ax.set_xticks(range(len(ALL_VARS)))
    ax.set_yticks(range(len(ALL_VARS)))
    ax.set_xticklabels(ALL_VARS, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(ALL_VARS, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # 在每格寫上相關係數值
    for i in range(len(ALL_VARS)):
        for j in range(len(ALL_VARS)):
            v = corr.iloc[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color=color, fontsize=9)

    # 顏色條
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_FOLDER, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=120, bbox_inches="tight")
plt.close()


# =========================================================
# 6. 輸出 CSV（給論文表格用）
# =========================================================
out_q = os.path.join(OUTPUT_FOLDER, "correlation_quarterly.csv")
out_y = os.path.join(OUTPUT_FOLDER, "correlation_yearly.csv")
corr_q.round(4).to_csv(out_q, encoding="utf-8-sig")
corr_y.round(4).to_csv(out_y, encoding="utf-8-sig")


# =========================================================
# 完成
# =========================================================
print("\n=== 完成 ===")
print(f"季版相關矩陣：{out_q}")
print(f"年版相關矩陣：{out_y}")
print(f"熱力圖：{heatmap_path}")
print("\n判讀指引：")
print(f"  |corr| >= {SEVERE_THRESHOLD}：嚴重共線，VIF 必爆，必須處理（剔除或合成）")
print(f"  |corr| >= {MODERATE_THRESHOLD}：中度共線，VIF 通常 5~10，建議處理")
print(f"  |corr| >= {WARN_THRESHOLD}：注意，VIF 可能介於 2~5，視情況處理")