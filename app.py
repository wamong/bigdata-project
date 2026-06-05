import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ──────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "report", "archive", "hybrid_student_performance_1200.csv")

# ── 순서형 매핑 ────────────────────────────────────────────────────
ORDINAL_MAPS = {
    "academic_satisfaction": {
        "Very unsatisfied": 1, "Unsatisfied": 2, "Neutral": 3, "Satisfied": 4, "Very satisfied": 5,
    },
    "study_hours_daily": {
        "Less than 1 hour": 0, "1–2 hours": 1, "More than 2 hours": 2,
    },
    "revision_frequency": {
        "Never": 0, "Rarely": 1, "Few times a week": 2, "Daily": 3,
    },
    "focus_duration": {
        "30–60 minutes": 0, "1–2 hours": 1, "More than 2 hours": 2,
    },
    "screen_time_non_study": {
        "2–4 hours": 0, "4–6 hours": 1, "More than 6 hours": 2,
    },
    "study_consistency": {
        "Rarely": 0, "Sometimes": 1, "Mostly consistent": 2,
    },
    "tasks_on_time": {
        "Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3,
    },
    "sleepy_during_study": {
        "Never": 0, "Sometimes": 1, "Often": 2, "Always": 3,
    },
    "sleep_hours": {
        "4–5 hours": 0, "6–7 hours": 1, "More than 8 hours": 2,
    },
    "assignments_on_time": {
        "Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3,
    },
    "attendance_percentage": {
        "Less than 50%": 0, "50% – 65%": 1, "66% – 75%": 2,
        "76% – 85%": 3, "Above 85%": 4,
    },
    "external_resources": {
        "Never (Unaware or Not interested)": 0,
        "Rarely (Passive)": 1,
        "Occasionally (When needed)": 2,
    },
    "external_pressure": {
        "No Impact (Fully supportive environment)": 0,
        "Low Impact (Rarely affects study)": 1,
        "Moderate Impact (Occasional disruption)": 2,
        "High Impact (Frequent disruption)": 3,
    },
    "career_goal_clarity": {
        "Not clear": 0, "Somewhat clear": 1, "Very clear": 2,
    },
    "programming_foundation": {
        "Limited knowledge, theoretical only": 0,
        "Basic knowledge, learning while practicing": 1,
        "Strong foundation in core concepts": 2,
    },
    "events_participation": {
        "Never participate in such events": 0,
        "Rarely participate, mostly observe": 1,
        "Occasionally participate in events": 2,
    },
    "performance_risk_level": {
        "Low Risk": 0, "Moderate Risk": 1, "High Risk": 2,
    },
    "cgpa_category": {
        "5.0 – 6.9": 0, "7.0 – 8.4": 1,
        "8.5 – 9.4": 2, "9.5 – 10.0": 3,
    },
}

OHE_COLS = [
    "year_class", "program_stream", "gender",
    "main_distractor", "skills_developing",
    "career_interest", "online_courses",
    "projects_internships", "preparation_status",
    "strongest_asset", "internal_barrier",
]

SCALE_COLS  = ["age", "daily_productivity", "energy_level", "stress_level", "routine_rating"]
TARGET_COLS = ["cgpa_category", "performance_risk_level"]
DROP_COLS   = ["student_id", "timestamp"]

CGPA_ORDER  = ["5.0 – 6.9", "7.0 – 8.4", "8.5 – 9.4", "9.5 – 10.0"]
RISK_ORDER  = ["Low Risk", "Moderate Risk", "High Risk"]
RISK_COLOR  = {"Low Risk": "#4CAF50", "Moderate Risk": "#FFC107", "High Risk": "#F44336"}
STUDY_ORDER = ["Less than 1 hour", "1–2 hours", "More than 2 hours"]
SLEEP_ORDER = ["4–5 hours", "6–7 hours", "More than 8 hours"]
ATT_ORDER   = ["Less than 50%", "50% – 65%", "66% – 75%", "76% – 85%", "Above 85%"]


# ── 데이터 로드 / 전처리 ───────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def preprocess(_df_raw: pd.DataFrame):
    df = _df_raw.drop(columns=[c for c in DROP_COLS if c in _df_raw.columns])

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    ohe_exist = [c for c in OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_exist, drop_first=False, dtype=int)

    scale_exist = [c for c in SCALE_COLS if c in df.columns]
    scaler = StandardScaler()
    df[scale_exist] = scaler.fit_transform(df[scale_exist])

    return df


def fig2st(fig):
    st.pyplot(fig)
    plt.close(fig)


# ── 앱 레이아웃 ────────────────────────────────────────────────────
st.set_page_config(
    page_title="학생 성취도 EDA & 전처리",
    page_icon="📚",
    layout="wide",
)

st.title("📚 학생 성취도 예측 — EDA & 전처리")
st.caption("20242530 정명진 | hybrid_student_performance_1200.csv")

df_raw     = load_raw()
cgpa_order = [c for c in CGPA_ORDER if c in df_raw["cgpa_category"].unique()]
risk_order = [r for r in RISK_ORDER  if r in df_raw["performance_risk_level"].unique()]

tab1, tab2, tab3 = st.tabs(["📋 데이터 개요", "🔍 EDA", "⚙️ 전처리"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — 데이터 개요
# ══════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("전체 행", df_raw.shape[0])
    col_b.metric("전체 컬럼", df_raw.shape[1])
    col_c.metric("결측치 있는 컬럼", int((df_raw.isnull().sum() > 0).sum()))
    col_d.metric("중복 행", int(df_raw.duplicated().sum()))

    st.subheader("샘플 데이터 (상위 5행)")
    st.dataframe(df_raw.head(5), use_container_width=True)

    st.subheader("컬럼 정보")
    info = pd.DataFrame({
        "타입":      df_raw.dtypes.astype(str),
        "결측치":    df_raw.isnull().sum(),
        "결측률(%)": (df_raw.isnull().sum() / len(df_raw) * 100).round(2),
        "유니크 수": df_raw.nunique(),
    })
    st.dataframe(info, use_container_width=True)

    st.subheader("수치형 기술통계")
    st.dataframe(
        df_raw.select_dtypes("number").describe().T.round(3),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════
with tab2:
    eda_section = st.radio(
        "분석 항목 선택",
        ["타겟 변수 분포", "수치형 변수", "범주형 변수", "상관관계 히트맵", "변수 × 타겟 관계"],
        horizontal=True,
    )
    st.divider()

    # ── 타겟 분포
    if eda_section == "타겟 변수 분포":
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        cgpa_cnt = df_raw["cgpa_category"].value_counts().reindex(cgpa_order, fill_value=0)
        bars = axes[0].bar(cgpa_cnt.index, cgpa_cnt.values,
                           color=sns.color_palette("Blues_d", len(cgpa_cnt)), edgecolor="white")
        for b in bars:
            axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                         int(b.get_height()), ha="center", fontsize=9)
        axes[0].set_title("CGPA 구간 분포 (회귀 타겟)", fontweight="bold")
        axes[0].set_xlabel("CGPA 구간")
        axes[0].set_ylabel("학생 수")
        axes[0].tick_params(axis="x", rotation=20)

        risk_cnt = df_raw["performance_risk_level"].value_counts().reindex(risk_order, fill_value=0)
        axes[1].pie(risk_cnt.values, labels=risk_cnt.index, autopct="%1.1f%%",
                    colors=[RISK_COLOR[r] for r in risk_order], startangle=140,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                    textprops={"fontsize": 11})
        axes[1].set_title("성과 위험도 분포 (분류 타겟)", fontweight="bold")

        plt.tight_layout()
        fig2st(fig)

    # ── 수치형 변수
    elif eda_section == "수치형 변수":
        num_feats = ["age", "daily_productivity", "energy_level",
                     "stress_level", "routine_rating", "academic_satisfaction"]
        plot_type = st.selectbox("그래프 유형", ["히스토그램", "박스플롯 (위험도별)"])

        if plot_type == "히스토그램":
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            pal = sns.color_palette("muted", len(num_feats))
            for i, col in enumerate(num_feats):
                data = df_raw[col].dropna()
                axes[i].hist(data, bins=20, color=pal[i], edgecolor="white", alpha=0.85)
                axes[i].axvline(data.mean(),   color="red",    linestyle="--", lw=1.4,
                                label=f"평균 {data.mean():.2f}")
                axes[i].axvline(data.median(), color="orange", linestyle=":",  lw=1.4,
                                label=f"중앙값 {data.median():.2f}")
                axes[i].set_title(col, fontweight="bold")
                axes[i].legend(fontsize=8)
            plt.suptitle("수치형 변수 히스토그램", fontsize=13, fontweight="bold", y=1.01)
            plt.tight_layout()
            fig2st(fig)
        else:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            for i, col in enumerate(num_feats):
                sns.boxplot(data=df_raw, x="performance_risk_level", y=col,
                            order=risk_order, palette=RISK_COLOR,
                            ax=axes[i], width=0.5)
                axes[i].set_title(f"{col} by 위험도", fontweight="bold")
                axes[i].set_xlabel("")
                axes[i].tick_params(axis="x", rotation=15)
            plt.suptitle("위험도별 수치형 변수 (박스플롯)", fontsize=13, fontweight="bold", y=1.01)
            plt.tight_layout()
            fig2st(fig)

    # ── 범주형 변수
    elif eda_section == "범주형 변수":
        cat_options = {
            "공부 시간 (study_hours_daily)":      ("study_hours_daily", STUDY_ORDER),
            "수면 시간 (sleep_hours)":             ("sleep_hours", SLEEP_ORDER),
            "출석률 (attendance_percentage)":      ("attendance_percentage", ATT_ORDER),
            "성별 (gender)":                       ("gender", None),
            "학년 (year_class)":                   ("year_class", None),
            "전공 트랙 (program_stream)":          ("program_stream", None),
            "주요 방해 요소 (main_distractor)":    ("main_distractor", None),
            "내부 장벽 (internal_barrier)":        ("internal_barrier", None),
        }
        selected = st.selectbox("컬럼 선택", list(cat_options.keys()))
        col, order = cat_options[selected]
        if order:
            order_f = [o for o in order if o in df_raw[col].unique()]
            counts   = df_raw[col].value_counts().reindex(order_f, fill_value=0)
        else:
            counts = df_raw[col].value_counts()

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(counts.index[::-1], counts.values[::-1],
                       color=sns.color_palette("pastel", len(counts)), edgecolor="white")
        for b in bars:
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                    int(b.get_width()), va="center", fontsize=9)
        ax.set_title(selected, fontsize=12, fontweight="bold")
        ax.set_xlabel("학생 수")
        ax.set_xlim(0, counts.max() * 1.14)
        plt.tight_layout()
        fig2st(fig)

    # ── 상관관계 히트맵
    elif eda_section == "상관관계 히트맵":
        num_cols = df_raw.select_dtypes(include="number").columns.tolist()
        corr     = df_raw[num_cols].corr()
        mask     = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, annot_kws={"size": 10}, ax=ax)
        ax.set_title("수치형 변수 상관관계 히트맵", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig2st(fig)

        pairs = (corr.where(~mask).stack().reset_index()
                 .rename(columns={"level_0": "변수1", "level_1": "변수2", 0: "상관계수"}))
        pairs = (pairs.assign(절댓값=pairs["상관계수"].abs())
                 .sort_values("절댓값", ascending=False)
                 .drop("절댓값", axis=1).head(8).reset_index(drop=True))
        st.caption("▶ 상관계수 절댓값 상위 8쌍")
        st.dataframe(pairs, use_container_width=True)

    # ── 변수 × 타겟 관계
    else:
        chart_options = {
            "출석률 × CGPA (교차 히트맵)":         "att_cgpa",
            "공부 시간 × CGPA (누적 막대)":         "study_cgpa",
            "스트레스 vs 에너지 산점도 (위험도별)": "scatter",
            "수면 시간 × 위험도 (카운트플롯)":      "sleep_risk",
            "CGPA별 생산성/스트레스 (바이올린)":    "violin",
            "시험 준비 상태 × 위험도 (비율 막대)":  "prep_risk",
        }
        chosen = st.selectbox("차트 선택", list(chart_options.keys()))
        key = chart_options[chosen]

        att_order   = [a for a in ATT_ORDER   if a in df_raw["attendance_percentage"].unique()]
        sleep_order = [s for s in SLEEP_ORDER  if s in df_raw["sleep_hours"].unique()]
        study_order = [s for s in STUDY_ORDER  if s in df_raw["study_hours_daily"].unique()]

        if key == "att_cgpa":
            cross = (pd.crosstab(df_raw["attendance_percentage"], df_raw["cgpa_category"])
                     .reindex(att_order, fill_value=0)
                     .reindex(columns=cgpa_order, fill_value=0))
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
            ax.set_title("출석률 × CGPA 구간 교차 빈도", fontweight="bold")
            plt.tight_layout()
            fig2st(fig)

        elif key == "study_cgpa":
            cross = (pd.crosstab(df_raw["study_hours_daily"], df_raw["cgpa_category"])
                     .reindex(study_order, fill_value=0)
                     .reindex(columns=cgpa_order, fill_value=0))
            cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            cross.plot(kind="bar", color=sns.color_palette("Blues", len(cgpa_order)),
                       ax=axes[0], edgecolor="white")
            axes[0].set_title("공부 시간 × CGPA (빈도)", fontweight="bold")
            axes[0].tick_params(axis="x", rotation=20)
            axes[0].legend(title="CGPA", fontsize=8, bbox_to_anchor=(1, 1))
            cross_pct.plot(kind="bar", stacked=True,
                           color=sns.color_palette("Blues", len(cgpa_order)),
                           ax=axes[1], edgecolor="white")
            axes[1].set_title("공부 시간 × CGPA (누적 비율)", fontweight="bold")
            axes[1].set_ylabel("비율 (%)")
            axes[1].tick_params(axis="x", rotation=20)
            axes[1].legend(title="CGPA", fontsize=8, bbox_to_anchor=(1, 1))
            plt.tight_layout()
            fig2st(fig)

        elif key == "scatter":
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for risk in risk_order:
                g = df_raw[df_raw["performance_risk_level"] == risk]
                axes[0].scatter(g["stress_level"], g["energy_level"],
                                c=RISK_COLOR[risk], label=risk, alpha=0.5, s=35,
                                edgecolors="white", linewidths=0.3)
            axes[0].set_title("스트레스 vs 에너지 (산점도)", fontweight="bold")
            axes[0].set_xlabel("스트레스 레벨")
            axes[0].set_ylabel("에너지 레벨")
            axes[0].legend(title="위험도")
            for risk in risk_order:
                g = df_raw[df_raw["performance_risk_level"] == risk]["stress_level"].dropna()
                sns.kdeplot(g, ax=axes[1], label=risk, color=RISK_COLOR[risk], fill=True, alpha=0.25)
            axes[1].set_title("스트레스 레벨 KDE (위험도별)", fontweight="bold")
            axes[1].set_xlabel("스트레스 레벨")
            axes[1].legend(title="위험도")
            plt.tight_layout()
            fig2st(fig)

        elif key == "sleep_risk":
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(data=df_raw, x="sleep_hours", hue="performance_risk_level",
                          order=sleep_order, hue_order=risk_order,
                          palette=RISK_COLOR, ax=ax, edgecolor="white")
            ax.set_title("수면 시간별 위험도 분포", fontsize=12, fontweight="bold")
            ax.set_xlabel("수면 시간")
            ax.set_ylabel("학생 수")
            ax.legend(title="위험도")
            plt.tight_layout()
            fig2st(fig)

        elif key == "violin":
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, col, pal in zip(axes,
                                    ["daily_productivity", "stress_level"],
                                    ["Blues", "Reds"]):
                sns.violinplot(data=df_raw, x="cgpa_category", y=col,
                               order=cgpa_order, palette=pal, ax=ax, cut=0, inner="quartile")
                ax.set_title(f"CGPA 구간별 {col}", fontweight="bold")
                ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            fig2st(fig)

        elif key == "prep_risk":
            prep_cross = pd.crosstab(df_raw["preparation_status"], df_raw["performance_risk_level"])
            prep_cross = prep_cross.reindex(columns=risk_order, fill_value=0)
            prep_pct   = prep_cross.div(prep_cross.sum(axis=1), axis=0) * 100
            fig, ax = plt.subplots(figsize=(10, 5))
            prep_pct.plot(kind="bar", stacked=True,
                          color=[RISK_COLOR[r] for r in risk_order],
                          ax=ax, edgecolor="white", width=0.65)
            ax.set_title("시험 준비 상태별 위험도 비율", fontsize=12, fontweight="bold")
            ax.set_xlabel("준비 상태")
            ax.set_ylabel("비율 (%)")
            ax.tick_params(axis="x", rotation=25)
            ax.legend(title="위험도", bbox_to_anchor=(1, 1))
            plt.tight_layout()
            fig2st(fig)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — 전처리
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("전처리 파이프라인")

    with st.expander("① 드롭 컬럼", expanded=False):
        st.markdown(f"식별자·메타 컬럼 제거: `{DROP_COLS}`")

    with st.expander("② 결측치 처리", expanded=False):
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        m_df = missing.reset_index()
        m_df.columns = ["컬럼", "결측 수"]
        m_df["결측률(%)"] = (m_df["결측 수"] / len(df_raw) * 100).round(2)
        m_df["처리 방법"] = m_df["컬럼"].apply(
            lambda c: "중앙값 대체" if df_raw[c].dtype != object else "최빈값 대체"
        )
        st.dataframe(m_df, use_container_width=True)

    with st.expander("③ Ordinal Encoding", expanded=False):
        rows = []
        for col, mapping in ORDINAL_MAPS.items():
            if col in df_raw.columns:
                rows.append({
                    "컬럼": col,
                    "범위": f"0 ~ {max(mapping.values())}",
                    "예시 (원본 → 숫자)": str(list(mapping.items())[:3]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("④ One-Hot Encoding", expanded=False):
        ohe_exist = [c for c in OHE_COLS if c in df_raw.columns]
        st.markdown("**대상 컬럼** (drop_first=False)")
        st.write(ohe_exist)

    with st.expander("⑤ StandardScaler", expanded=False):
        scale_exist = [c for c in SCALE_COLS if c in df_raw.columns]
        st.markdown(f"**대상 컬럼**: `{scale_exist}`")
        st.markdown(f"타겟 컬럼 `{TARGET_COLS}` 은 스케일링 제외")

    st.divider()

    if st.button("▶ 전처리 실행", type="primary"):
        with st.spinner("전처리 중..."):
            df_proc = preprocess(df_raw)

        st.success(
            f"완료: {df_proc.shape[0]}행 × {df_proc.shape[1]}열  |  "
            f"결측치 {df_proc.isnull().sum().sum()}개"
        )

        col1, col2 = st.columns(2)
        col1.metric("원본 컬럼 수", df_raw.shape[1])
        col2.metric("전처리 후 컬럼 수", df_proc.shape[1],
                    delta=f"+{df_proc.shape[1] - df_raw.shape[1]}")

        st.subheader("전처리 결과 미리보기")
        st.dataframe(df_proc.head(10), use_container_width=True)

        st.subheader("타겟 클래스 분포 (전처리 후)")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, col, title in zip(
            axes,
            ["cgpa_category", "performance_risk_level"],
            ["CGPA 구간 (0~3)", "위험도 (0=Low, 1=Moderate, 2=High)"],
        ):
            cnt  = df_proc[col].value_counts().sort_index()
            bars = ax.bar([str(x) for x in cnt.index], cnt.values,
                          color=sns.color_palette("muted", len(cnt)), edgecolor="white")
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                        int(b.get_height()), ha="center", fontsize=9)
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("학생 수")
        plt.tight_layout()
        fig2st(fig)

        csv = df_proc.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇ 전처리 데이터 CSV 다운로드",
            data=csv,
            file_name="preprocessed_student_performance.csv",
            mime="text/csv",
        )
