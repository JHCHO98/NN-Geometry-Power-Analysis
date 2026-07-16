import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. open()을 사용하여 CSV 파일 입력받기
file_path = "nn_geometry_power_log.csv"

with open(file_path, "r", encoding="utf-8") as f:
    # open()으로 연 파일 객체를 pandas가 바로 읽도록 처리
    df = pd.read_csv(f)

# 2. 데이터 표준화 및 파생 변수 계산
# Loop_Count가 모델 전체(-1)와 개별 레이어가 다르므로, 단위 루프(1회 추론)당 지표로 변환
df["Duration_per_Loop_Sec"] = df["Duration_Sec"] / df["Loop_Count"]
df["Energy_per_Loop_J"] = (
    df["Energy_Consumed_kWh"] * 3600000
) / df["Loop_Count"]  # kWh를 Joule로 변환

# 단위 시간당 소모 전력 (W = J / Sec) 계산
df["Average_Power_Watts"] = (
    df["Energy_Consumed_kWh"] * 3600000
) / df["Duration_Sec"]

# 데이터셋 분리 (전체 모델 vs 개별 레이어 파편)
full_model_df = df[df["Layer_Index"] == -1].copy()
layer_df = df[df["Layer_Index"] != -1].copy()

# 콘솔창에 거시적 분석 결과 요약 출력
print("=== [거시적 분석] 6종 모델별 전체 추론 성능 및 전력 지표 ===")
print(
    full_model_df[
        [
            "Model_Mode",
            "Duration_Sec",
            "Energy_Consumed_kWh",
            "Energy_per_Loop_J",
            "Average_Power_Watts",
        ]
    ].to_string(index=False)
)
print("\n" + "=" * 60 + "\n")


# 3. 시각화 환경 및 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"  # 한글 깨짐 방지가 필요하면 고딕 계열 설정
plt.rcParams["axes.unicode_minus"] = False

# --- 그래프 1: 전체 모델별 단위 추론당 에너지 소모량 (Joule) ---
plt.figure(figsize=(10, 6))
barplot1 = sns.barplot(
    data=full_model_df.sort_values(by="Energy_per_Loop_J", ascending=False),
    x="Model_Mode",
    y="Energy_per_Loop_J",
    palette="viridis",
)
plt.title(
    "Overall Energy Consumption per Inference by Model Geometry",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Model Geometric Configuration", fontsize=12)
plt.ylabel("Energy consumed per Inference (Joule)", fontsize=12)
plt.xticks(rotation=15)
for p in barplot1.patches:
    barplot1.annotate(
        f"{p.get_height():.4f} J",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 9),
        textcoords="offset points",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig("overall_energy_comparison.png", dpi=300)
plt.show()

# --- 그래프 2: 연산 집약형(Conv2d) vs 메모리 집약형(Linear) 레이어별 평균 전력 비교 ---
target_layers = layer_df[layer_df["Layer_Type"].isin(["Conv2d", "Linear"])]
plt.figure(figsize=(9, 6))
sns.boxplot(
    data=target_layers,
    x="Layer_Type",
    y="Average_Power_Watts",
    palette="Set2",
    hue="Layer_Type",
    legend=False,
)
sns.stripplot(data=target_layers, x="Layer_Type", y="Average_Power_Watts", color="black", alpha=0.5)
plt.title(
    "Hardware Power Demand: Compute-bound (Conv2d) vs Memory-bound (Linear)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Layer Operation Type", fontsize=12)
plt.ylabel("Average Power Demand (Watts)", fontsize=12)
plt.tight_layout()
plt.savefig("layer_type_power_bottleneck.png", dpi=300)
plt.show()

# --- 그래프 3: 레이어 전개에 따른 동적 전력 소모 흐름 분석 (Deep & Narrow vs Uniform)
compare_modes = ["deep_narrow", "uniform", "mid_balanced", "shallow_wide", "funnel_wide_to_narrow", "hourglass"]
colors=["red","orange","purple","green","gold","skyblue"]

for (mode,color) in zip(compare_modes,colors):
    flow_df = layer_df[layer_df["Model_Mode"].isin([mode])]
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=flow_df,
        x="Layer_Index",
        y="Energy_per_Loop_J",
        marker="o",
        linewidth=2,
        color=color
    )
    plt.title(
        f"Dynamic Energy Consumption Flow Across Layer Progression ({mode})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Layer Index (Sequence)", fontsize=12)
    plt.ylabel("Energy per Burst Loop (Joule)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"dynamic_energy_flow_{mode}.png", dpi=300)
    plt.show()