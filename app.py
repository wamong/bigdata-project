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

# ── 경로 ──────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "report", "archive", "hybrid_student_performance_1200.csv")

# ══════════════════════════════════════════════════════════════════
# 번역 사전
# ══════════════════════════════════════════════════════════════════
COL_KO = {
    "student_id": "학번", "timestamp": "제출 시각",
    "age": "나이", "gender": "성별",
    "year_class": "학년", "program_stream": "전공 트랙",
    "cgpa_category": "CGPA 구간",
    "academic_satisfaction": "학업 만족도",
    "study_hours_daily": "일일 공부 시간",
    "daily_productivity": "일일 생산성",
    "revision_frequency": "복습 빈도",
    "focus_duration": "집중 지속 시간",
    "screen_time_non_study": "비학습 스크린 타임",
    "main_distractor": "주요 방해 요소",
    "study_consistency": "공부 일관성",
    "tasks_on_time": "과제 제때 제출",
    "preparation_status": "시험 준비 상태",
    "career_goal_clarity": "진로 목표 명확도",
    "skills_developing": "개발 역량 유형",
    "energy_level": "에너지 레벨",
    "stress_level": "스트레스 레벨",
    "routine_rating": "루틴 평가",
    "sleepy_during_study": "공부 중 졸림",
    "sleep_hours": "수면 시간",
    "career_interest": "진로 관심사",
    "online_courses": "온라인 강좌",
    "projects_internships": "프로젝트/인턴십",
    "programming_foundation": "프로그래밍 기초",
    "events_participation": "행사 참여",
    "assignments_on_time": "과제 제출",
    "attendance_percentage": "출석률",
    "strongest_asset": "주요 강점",
    "internal_barrier": "내부 장벽",
    "external_resources": "외부 자료 활용",
    "external_pressure": "외부 압박",
    "performance_risk_level": "학업 관리 단계",
}

VAL_KO = {
    "gender": {"Female": "여성", "Male": "남성", "Other": "기타"},
    "year_class": {
        "First Year (FY)": "1학년", "Second Year (SY)": "2학년",
        "Third Year (TY)": "3학년", "Final Year": "4학년",
        "First Year (PG)": "대학원 1학년",
    },
    "program_stream": {
        "BCA": "BCA", "BCom": "BCom",
        "BSc Cyber Security": "사이버보안", "BSc IT": "IT학과",
        "BA": "인문학과", "BSc CS": "컴퓨터과학", "BBA": "경영학과",
    },
    "cgpa_category": {
        "5.0 – 6.9": "5.0~6.9", "7.0 – 8.4": "7.0~8.4",
        "8.5 – 9.4": "8.5~9.4", "9.5 – 10.0": "9.5~10.0",
    },
    "academic_satisfaction": {
        "Very unsatisfied": "매우 불만족", "Unsatisfied": "불만족",
        "Neutral": "보통", "Satisfied": "만족", "Very satisfied": "매우 만족",
    },
    "study_hours_daily": {
        "Less than 1 hour": "1시간 미만", "1–2 hours": "1~2시간",
        "More than 2 hours": "2시간 초과",
    },
    "revision_frequency": {
        "Never": "안 함", "Rarely": "거의 안 함",
        "Few times a week": "주 몇 번", "Daily": "매일",
    },
    "focus_duration": {
        "30–60 minutes": "30~60분", "1–2 hours": "1~2시간",
        "More than 2 hours": "2시간 초과",
    },
    "screen_time_non_study": {
        "2–4 hours": "2~4시간", "4–6 hours": "4~6시간",
        "More than 6 hours": "6시간 초과",
    },
    "main_distractor": {
        "Social media": "SNS",
        "Video content (YouTube/OTT)": "유튜브/OTT",
        "Social interactions": "친구/사교",
        "Gaming": "게임", "Other": "기타",
    },
    "study_consistency": {
        "Rarely": "거의 안 함", "Sometimes": "가끔",
        "Mostly consistent": "대체로 일정",
    },
    "tasks_on_time": {
        "Rarely": "거의 안 함", "Sometimes": "가끔",
        "Often": "자주", "Always": "항상",
    },
    "preparation_status": {
        "Planning to start soon": "곧 시작 예정",
        "Actively preparing for a goal (placements/exams)": "적극 준비 중",
        "Thinking about it": "고민 중",
    },
    "career_goal_clarity": {
        "Not clear": "불명확", "Somewhat clear": "어느 정도 명확",
        "Very clear": "매우 명확",
    },
    "skills_developing": {
        "Hard skills (programming, data analytics, technical skills)": "하드 스킬",
        "Both hard and soft skills": "하드+소프트 스킬",
        "Soft skills (communication, teamwork, leadership, financial literacy)": "소프트 스킬",
    },
    "sleepy_during_study": {
        "Never": "안 졸림", "Sometimes": "가끔",
        "Often": "자주", "Always": "항상",
    },
    "sleep_hours": {
        "4–5 hours": "4~5시간", "6–7 hours": "6~7시간",
        "More than 8 hours": "8시간 초과",
    },
    "career_interest": {
        "Other": "기타", "Automation Engineer": "자동화 엔지니어",
        "Cyber Security Analyst": "사이버보안 분석가",
        "Data Analyst": "데이터 분석가", "AI / ML": "AI/ML",
        "Software Developer": "소프트웨어 개발자",
        "Web Developer": "웹 개발자",
    },
    "online_courses": {
        "Not currently, but intend to in the future": "미래 계획 있음",
        "Yes, currently enrolled in one or more courses/certifications": "현재 수강 중",
        "Planning to enroll soon": "곧 등록 예정",
        "No, not interested": "관심 없음",
    },
    "projects_internships": {
        "Yes, actively working on projects/internship": "적극 참여 중",
        "Planning to start a project/internship soon": "곧 시작 예정",
        "Not currently, but intend to in the future": "미래 계획 있음",
    },
    "programming_foundation": {
        "Limited knowledge, theoretical only": "이론적 지식만",
        "Basic knowledge, learning while practicing": "기본 · 실습 중",
        "Strong foundation in core concepts": "핵심 개념 탄탄",
    },
    "events_participation": {
        "Never participate in such events": "참여 안 함",
        "Rarely participate, mostly observe": "관찰 위주",
        "Occasionally participate in events": "가끔 참여",
    },
    "assignments_on_time": {
        "Rarely": "거의 안 함", "Sometimes": "가끔",
        "Often": "자주", "Always": "항상",
    },
    "attendance_percentage": {
        "Less than 50%": "50% 미만", "50% – 65%": "50~65%",
        "66% – 75%": "66~75%", "76% – 85%": "76~85%",
        "Above 85%": "85% 초과",
    },
    "strongest_asset": {
        "Technical/Hard Skills (Coding, Math, Logic)": "기술적 역량",
        "Creative/Design Skills (Innovation, UI/UX, Content)": "창의/디자인",
        "Management/Execution (Planning, Organizing, Discipline)": "관리/실행력",
        "Soft Skills (Communication, Leadership, Teamwork)": "소프트 스킬",
    },
    "internal_barrier": {
        "Lack of Consistency or Determination (Difficulty sticking to a plan)": "의지/지속성 부족",
        "Difficulty with Focus / Concentration": "집중력 부족",
        "Procrastination / Low Motivation": "미루는 습관",
        "Poor Time Management / Over-scheduling": "시간 관리 미흡",
    },
    "external_resources": {
        "Never (Unaware or Not interested)": "전혀 안 함",
        "Rarely (Passive)": "거의 안 함",
        "Occasionally (When needed)": "가끔",
    },
    "external_pressure": {
        "No Impact (Fully supportive environment)": "영향 없음",
        "Low Impact (Rarely affects study)": "낮은 영향",
        "Moderate Impact (Occasional disruption)": "보통 영향",
        "High Impact (Frequent disruption)": "높은 영향",
    },
    "performance_risk_level": {
        "Low Risk": "안정", "Moderate Risk": "주의 관찰", "High Risk": "관리 필요",
    },
}

# ── 한국어 순서 리스트 ─────────────────────────────────────────────
CGPA_ORDER_EN  = ["5.0 – 6.9", "7.0 – 8.4", "8.5 – 9.4", "9.5 – 10.0"]
RISK_ORDER_EN  = ["Low Risk", "Moderate Risk", "High Risk"]
STUDY_ORDER_EN = ["Less than 1 hour", "1–2 hours", "More than 2 hours"]
SLEEP_ORDER_EN = ["4–5 hours", "6–7 hours", "More than 8 hours"]
ATT_ORDER_EN   = ["Less than 50%", "50% – 65%", "66% – 75%", "76% – 85%", "Above 85%"]
SAT_ORDER_EN   = ["Very unsatisfied", "Unsatisfied", "Neutral", "Satisfied", "Very satisfied"]

CGPA_ORDER_KO  = [VAL_KO["cgpa_category"][v] for v in CGPA_ORDER_EN]
RISK_ORDER_KO  = [VAL_KO["performance_risk_level"][v] for v in RISK_ORDER_EN]
STUDY_ORDER_KO = [VAL_KO["study_hours_daily"][v] for v in STUDY_ORDER_EN]
SLEEP_ORDER_KO = [VAL_KO["sleep_hours"][v] for v in SLEEP_ORDER_EN]
ATT_ORDER_KO   = [VAL_KO["attendance_percentage"][v] for v in ATT_ORDER_EN]
SAT_ORDER_KO   = [VAL_KO["academic_satisfaction"][v] for v in SAT_ORDER_EN]

RISK_COLOR_KO = {"안정": "#4CAF50", "주의 관찰": "#FFC107", "관리 필요": "#F44336"}

# ── 전처리 설정 ────────────────────────────────────────────────────
ORDINAL_MAPS = {
    "academic_satisfaction": {
        "Very unsatisfied": 1, "Unsatisfied": 2, "Neutral": 3,
        "Satisfied": 4, "Very satisfied": 5,
    },
    "study_hours_daily":    {"Less than 1 hour": 0, "1–2 hours": 1, "More than 2 hours": 2},
    "revision_frequency":   {"Never": 0, "Rarely": 1, "Few times a week": 2, "Daily": 3},
    "focus_duration":       {"30–60 minutes": 0, "1–2 hours": 1, "More than 2 hours": 2},
    "screen_time_non_study":{"2–4 hours": 0, "4–6 hours": 1, "More than 6 hours": 2},
    "study_consistency":    {"Rarely": 0, "Sometimes": 1, "Mostly consistent": 2},
    "tasks_on_time":        {"Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3},
    "sleepy_during_study":  {"Never": 0, "Sometimes": 1, "Often": 2, "Always": 3},
    "sleep_hours":          {"4–5 hours": 0, "6–7 hours": 1, "More than 8 hours": 2},
    "assignments_on_time":  {"Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3},
    "attendance_percentage": {
        "Less than 50%": 0, "50% – 65%": 1, "66% – 75%": 2,
        "76% – 85%": 3, "Above 85%": 4,
    },
    "external_resources": {
        "Never (Unaware or Not interested)": 0,
        "Rarely (Passive)": 1, "Occasionally (When needed)": 2,
    },
    "external_pressure": {
        "No Impact (Fully supportive environment)": 0,
        "Low Impact (Rarely affects study)": 1,
        "Moderate Impact (Occasional disruption)": 2,
        "High Impact (Frequent disruption)": 3,
    },
    "career_goal_clarity":    {"Not clear": 0, "Somewhat clear": 1, "Very clear": 2},
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
    "performance_risk_level": {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2},
    "cgpa_category": {
        "5.0 – 6.9": 0, "7.0 – 8.4": 1, "8.5 – 9.4": 2, "9.5 – 10.0": 3,
    },
}
OHE_COLS    = ["year_class", "program_stream", "gender", "main_distractor",
               "skills_developing", "career_interest", "online_courses",
               "projects_internships", "preparation_status", "strongest_asset", "internal_barrier"]
SCALE_COLS  = ["age", "daily_productivity", "energy_level", "stress_level", "routine_rating"]
TARGET_COLS = ["cgpa_category", "performance_risk_level"]
DROP_COLS   = ["student_id", "timestamp"]


# ── 유틸 함수 ──────────────────────────────────────────────────────
def ko(col: str) -> str:
    """컬럼명을 한국어로 반환."""
    return COL_KO.get(col, col)


def tval(val, col: str) -> str:
    """단일 값을 한국어로 반환."""
    return VAL_KO.get(col, {}).get(str(val), str(val))


def tdf(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """지정 컬럼의 값을 한국어로 번역한 복사본 반환."""
    d = df.copy()
    for c in cols:
        if c in VAL_KO:
            d[c] = d[c].map(lambda x: VAL_KO[c].get(str(x), str(x)))
    return d


def fig2st(fig):
    st.pyplot(fig)
    plt.close(fig)


def ko_order(order_en: list, col: str) -> list:
    """영어 순서 리스트를 한국어로 변환 (데이터에 존재하는 값만)."""
    return [VAL_KO[col][v] for v in order_en if v in VAL_KO.get(col, {})]


# ── 데이터 로드 / 전처리 ───────────────────────────────────────────
@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def preprocess(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    for c in df.select_dtypes("number").columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].fillna(df[c].mode()[0])
    for c, mapping in ORDINAL_MAPS.items():
        if c in df.columns:
            df[c] = df[c].map(mapping)
    ohe = [c for c in OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe, drop_first=False, dtype=int)
    sc = [c for c in SCALE_COLS if c in df.columns]
    df[sc] = StandardScaler().fit_transform(df[sc])
    return df


# ══════════════════════════════════════════════════════════════════
# 앱 시작
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="학생 성취도 EDA & 전처리", page_icon="📚", layout="wide")
st.title("📚 학생 성취도 예측 — EDA & 전처리")
st.caption("20242530 정명진 | hybrid_student_performance_1200.csv")

df_raw = load_raw()

# 실제 데이터에 존재하는 값만 추린 한국어 순서
cgpa_ko  = [VAL_KO["cgpa_category"][v]        for v in CGPA_ORDER_EN  if v in df_raw["cgpa_category"].values]
risk_ko  = [VAL_KO["performance_risk_level"][v] for v in RISK_ORDER_EN  if v in df_raw["performance_risk_level"].values]
study_ko = [VAL_KO["study_hours_daily"][v]     for v in STUDY_ORDER_EN if v in df_raw["study_hours_daily"].values]
sleep_ko = [VAL_KO["sleep_hours"][v]           for v in SLEEP_ORDER_EN if v in df_raw["sleep_hours"].dropna().values]
att_ko   = [VAL_KO["attendance_percentage"][v] for v in ATT_ORDER_EN   if v in df_raw["attendance_percentage"].values]
sat_ko   = [VAL_KO["academic_satisfaction"][v] for v in SAT_ORDER_EN   if v in df_raw["academic_satisfaction"].values]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 데이터 개요", "🔍 EDA", "⚙️ 전처리", "🧪 심화 분석", "🎛️ 필터 분석", "📊 발표 슬라이드"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — 데이터 개요
# ══════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 행", df_raw.shape[0])
    c2.metric("전체 컬럼", df_raw.shape[1])
    c3.metric("결측치 있는 컬럼", int((df_raw.isnull().sum() > 0).sum()))
    c4.metric("중복 행", int(df_raw.duplicated().sum()))

    st.subheader("샘플 데이터 (상위 5행)")
    st.dataframe(df_raw.head(5), use_container_width=True)

    st.subheader("컬럼 정보")
    info = pd.DataFrame({
        "컬럼명(원본)": df_raw.columns,
        "한국어 컬럼명": [COL_KO.get(c, c) for c in df_raw.columns],
        "타입":          df_raw.dtypes.values.astype(str),
        "결측치":        df_raw.isnull().sum().values,
        "결측률(%)":     (df_raw.isnull().sum().values / len(df_raw) * 100).round(2),
        "유니크 수":     df_raw.nunique().values,
    })
    st.dataframe(info, use_container_width=True)

    st.subheader("수치형 기술통계")
    num_desc = df_raw.select_dtypes("number").describe().T.round(3)
    num_desc.index = [COL_KO.get(c, c) for c in num_desc.index]
    st.dataframe(num_desc, use_container_width=True)


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

    # ── 타겟 분포 ──────────────────────────────────────────────────
    if eda_section == "타겟 변수 분포":
        df_t = tdf(df_raw, ["cgpa_category", "performance_risk_level"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        cgpa_cnt = df_t["cgpa_category"].value_counts().reindex(cgpa_ko, fill_value=0)
        bars = axes[0].bar(cgpa_cnt.index, cgpa_cnt.values,
                           color=sns.color_palette("Blues_d", len(cgpa_cnt)), edgecolor="white")
        for b in bars:
            axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                         int(b.get_height()), ha="center", fontsize=9)
        axes[0].set_title("CGPA 구간 분포 (회귀 타겟)", fontweight="bold")
        axes[0].set_xlabel("CGPA 구간")
        axes[0].set_ylabel("학생 수")
        axes[0].tick_params(axis="x", rotation=15)

        risk_cnt = df_t["performance_risk_level"].value_counts().reindex(risk_ko, fill_value=0)
        axes[1].pie(risk_cnt.values, labels=risk_cnt.index, autopct="%1.1f%%",
                    colors=[RISK_COLOR_KO[r] for r in risk_ko],
                    startangle=140,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                    textprops={"fontsize": 11})
        axes[1].set_title("학업 관리 단계 분포 (분류 타겟)", fontweight="bold")
        plt.tight_layout()
        fig2st(fig)

    # ── 수치형 변수 ────────────────────────────────────────────────
    elif eda_section == "수치형 변수":
        NUM_FEATS = ["age", "daily_productivity", "energy_level", "stress_level", "routine_rating"]
        plot_type = st.selectbox("그래프 유형", ["히스토그램", "박스플롯 (위험도별)"])

        if plot_type == "히스토그램":
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            pal = sns.color_palette("muted", len(NUM_FEATS))
            for i, col in enumerate(NUM_FEATS):
                data = df_raw[col].dropna()
                axes[i].hist(data, bins=20, color=pal[i], edgecolor="white", alpha=0.85)
                axes[i].axvline(data.mean(),   color="red",    linestyle="--", lw=1.4,
                                label=f"평균 {data.mean():.2f}")
                axes[i].axvline(data.median(), color="orange", linestyle=":",  lw=1.4,
                                label=f"중앙값 {data.median():.2f}")
                axes[i].set_title(ko(col), fontweight="bold")
                axes[i].set_xlabel("값")
                axes[i].set_ylabel("빈도")
                axes[i].legend(fontsize=8)
            axes[-1].axis("off")  # 6번째 칸 비움
            plt.suptitle("수치형 변수 히스토그램", fontsize=13, fontweight="bold", y=1.01)
            plt.tight_layout()
            fig2st(fig)
        else:
            df_b = tdf(df_raw, ["performance_risk_level"])
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            for i, col in enumerate(NUM_FEATS):
                sns.boxplot(data=df_b, x="performance_risk_level", y=col,
                            order=risk_ko, palette=RISK_COLOR_KO,
                            ax=axes[i], width=0.5)
                axes[i].set_title(f"{ko(col)} by 위험도", fontweight="bold")
                axes[i].set_xlabel("학업 관리 단계")
                axes[i].set_ylabel(ko(col))
                axes[i].tick_params(axis="x", rotation=15)
            axes[-1].axis("off")
            plt.suptitle("위험도별 수치형 변수 (박스플롯)", fontsize=13, fontweight="bold", y=1.01)
            plt.tight_layout()
            fig2st(fig)

    # ── 범주형 변수 ────────────────────────────────────────────────
    elif eda_section == "범주형 변수":
        cat_options = {
            "일일 공부 시간":      ("study_hours_daily",     STUDY_ORDER_EN),
            "수면 시간":           ("sleep_hours",           SLEEP_ORDER_EN),
            "출석률":              ("attendance_percentage", ATT_ORDER_EN),
            "학업 만족도":         ("academic_satisfaction", SAT_ORDER_EN),
            "성별":                ("gender",               None),
            "학년":                ("year_class",           None),
            "전공 트랙":           ("program_stream",       None),
            "주요 방해 요소":      ("main_distractor",      None),
            "내부 장벽":           ("internal_barrier",     None),
            "주요 강점":           ("strongest_asset",      None),
            "공부 중 졸림":        ("sleepy_during_study",  None),
            "복습 빈도":           ("revision_frequency",   None),
        }
        selected = st.selectbox("컬럼 선택", list(cat_options.keys()))
        col, order_en = cat_options[selected]
        df_c = tdf(df_raw, [col])

        if order_en:
            order_ko = [VAL_KO[col][v] for v in order_en if v in df_raw[col].unique()]
        else:
            order_ko = df_c[col].value_counts().index.tolist()

        counts = df_c[col].value_counts().reindex(order_ko, fill_value=0)

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(counts.index[::-1], counts.values[::-1],
                       color=sns.color_palette("pastel", len(counts)), edgecolor="white")
        for b in bars:
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                    int(b.get_width()), va="center", fontsize=9)
        ax.set_title(f"{selected} 분포", fontsize=12, fontweight="bold")
        ax.set_xlabel("학생 수")
        ax.set_xlim(0, counts.max() * 1.15)
        plt.tight_layout()
        fig2st(fig)

    # ── 상관관계 히트맵 ───────────────────────────────────────────
    elif eda_section == "상관관계 히트맵":
        num_cols = df_raw.select_dtypes("number").columns.tolist()
        corr     = df_raw[num_cols].corr()
        corr.columns = [ko(c) for c in corr.columns]
        corr.index   = [ko(c) for c in corr.index]
        mask = np.triu(np.ones_like(corr, dtype=bool))

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

    # ── 변수 × 타겟 관계 ──────────────────────────────────────────
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
        key    = chart_options[chosen]

        if key == "att_cgpa":
            df_c = tdf(df_raw, ["attendance_percentage", "cgpa_category"])
            cross = (pd.crosstab(df_c["attendance_percentage"], df_c["cgpa_category"])
                     .reindex(att_ko, fill_value=0)
                     .reindex(columns=cgpa_ko, fill_value=0))
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
            ax.set_title("출석률 × CGPA 구간 교차 빈도", fontweight="bold")
            ax.set_xlabel("CGPA 구간")
            ax.set_ylabel("출석률")
            plt.tight_layout()
            fig2st(fig)

        elif key == "study_cgpa":
            df_c = tdf(df_raw, ["study_hours_daily", "cgpa_category"])
            cross = (pd.crosstab(df_c["study_hours_daily"], df_c["cgpa_category"])
                     .reindex(study_ko, fill_value=0)
                     .reindex(columns=cgpa_ko, fill_value=0))
            cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            cross.plot(kind="bar", color=sns.color_palette("Blues", len(cgpa_ko)),
                       ax=axes[0], edgecolor="white")
            axes[0].set_title("공부 시간 × CGPA 구간 (빈도)", fontweight="bold")
            axes[0].set_xlabel("일일 공부 시간")
            axes[0].set_ylabel("학생 수")
            axes[0].tick_params(axis="x", rotation=20)
            axes[0].legend(title="CGPA 구간", fontsize=8, bbox_to_anchor=(1, 1))
            cross_pct.plot(kind="bar", stacked=True,
                           color=sns.color_palette("Blues", len(cgpa_ko)),
                           ax=axes[1], edgecolor="white")
            axes[1].set_title("공부 시간 × CGPA 구간 (누적 비율)", fontweight="bold")
            axes[1].set_xlabel("일일 공부 시간")
            axes[1].set_ylabel("비율 (%)")
            axes[1].tick_params(axis="x", rotation=20)
            axes[1].legend(title="CGPA 구간", fontsize=8, bbox_to_anchor=(1, 1))
            plt.tight_layout()
            fig2st(fig)

        elif key == "scatter":
            df_c = tdf(df_raw, ["performance_risk_level"])
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for risk in risk_ko:
                g = df_c[df_c["performance_risk_level"] == risk]
                axes[0].scatter(g["stress_level"], g["energy_level"],
                                c=RISK_COLOR_KO[risk], label=risk,
                                alpha=0.5, s=35, edgecolors="white", linewidths=0.3)
            axes[0].set_title("스트레스 vs 에너지 레벨 (산점도)", fontweight="bold")
            axes[0].set_xlabel("스트레스 레벨")
            axes[0].set_ylabel("에너지 레벨")
            axes[0].legend(title="학업 관리 단계")
            for risk in risk_ko:
                g = df_c[df_c["performance_risk_level"] == risk]["stress_level"].dropna()
                sns.kdeplot(g, ax=axes[1], label=risk, color=RISK_COLOR_KO[risk],
                            fill=True, alpha=0.25)
            axes[1].set_title("스트레스 레벨 밀도 (위험도별)", fontweight="bold")
            axes[1].set_xlabel("스트레스 레벨")
            axes[1].set_ylabel("밀도")
            axes[1].legend(title="학업 관리 단계")
            plt.tight_layout()
            fig2st(fig)

        elif key == "sleep_risk":
            df_c = tdf(df_raw, ["sleep_hours", "performance_risk_level"])
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(data=df_c, x="sleep_hours", hue="performance_risk_level",
                          order=sleep_ko, hue_order=risk_ko,
                          palette=RISK_COLOR_KO, ax=ax, edgecolor="white")
            ax.set_title("수면 시간별 위험도 분포", fontsize=12, fontweight="bold")
            ax.set_xlabel("수면 시간")
            ax.set_ylabel("학생 수")
            ax.legend(title="학업 관리 단계")
            plt.tight_layout()
            fig2st(fig)

        elif key == "violin":
            df_c = tdf(df_raw, ["cgpa_category"])
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, col, pal in zip(axes,
                                    ["daily_productivity", "stress_level"],
                                    ["Blues", "Reds"]):
                sns.violinplot(data=df_c, x="cgpa_category", y=col,
                               order=cgpa_ko, palette=pal, ax=ax, cut=0, inner="quartile")
                ax.set_title(f"CGPA 구간별 {ko(col)}", fontweight="bold")
                ax.set_xlabel("CGPA 구간")
                ax.set_ylabel(ko(col))
                ax.tick_params(axis="x", rotation=15)
            plt.tight_layout()
            fig2st(fig)

        elif key == "prep_risk":
            df_c = tdf(df_raw, ["preparation_status", "performance_risk_level"])
            prep_cross = pd.crosstab(df_c["preparation_status"],
                                     df_c["performance_risk_level"])
            prep_cross = prep_cross.reindex(columns=risk_ko, fill_value=0)
            prep_pct   = prep_cross.div(prep_cross.sum(axis=1), axis=0) * 100
            fig, ax = plt.subplots(figsize=(10, 5))
            prep_pct.plot(kind="bar", stacked=True,
                          color=[RISK_COLOR_KO[r] for r in risk_ko],
                          ax=ax, edgecolor="white", width=0.65)
            ax.set_title("시험 준비 상태별 위험도 비율", fontsize=12, fontweight="bold")
            ax.set_xlabel("시험 준비 상태")
            ax.set_ylabel("비율 (%)")
            ax.tick_params(axis="x", rotation=20)
            ax.legend(title="학업 관리 단계", bbox_to_anchor=(1, 1))
            plt.tight_layout()
            fig2st(fig)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — 전처리
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("전처리 파이프라인")

    with st.expander("① 불필요 컬럼 드롭", expanded=False):
        rows = [{"컬럼(원본)": c, "한국어 이름": COL_KO.get(c, c), "드롭 이유": "식별자/메타 정보"} for c in DROP_COLS]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("② 결측치 처리", expanded=False):
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        m_df = missing.reset_index()
        m_df.columns = ["컬럼(원본)", "결측 수"]
        m_df["한국어 컬럼명"] = m_df["컬럼(원본)"].map(lambda c: COL_KO.get(c, c))
        m_df["결측률(%)"]    = (m_df["결측 수"] / len(df_raw) * 100).round(2)
        m_df["처리 방법"]    = m_df["컬럼(원본)"].apply(
            lambda c: "중앙값 대체" if df_raw[c].dtype != object else "최빈값 대체"
        )
        st.dataframe(m_df[["컬럼(원본)", "한국어 컬럼명", "결측 수", "결측률(%)", "처리 방법"]],
                     use_container_width=True)

    with st.expander("③ Ordinal Encoding (순서형 → 정수)", expanded=False):
        rows = []
        for col, mapping in ORDINAL_MAPS.items():
            if col in df_raw.columns:
                sample = " | ".join(
                    f"{VAL_KO.get(col, {}).get(k, k)} → {v}"
                    for k, v in list(mapping.items())[:3]
                )
                rows.append({
                    "컬럼(원본)": col,
                    "한국어 컬럼명": COL_KO.get(col, col),
                    "범위": f"0 ~ {max(mapping.values())}",
                    "변환 예시": sample,
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("④ One-Hot Encoding (명목형)", expanded=False):
        ohe_rows = [{"컬럼(원본)": c, "한국어 컬럼명": COL_KO.get(c, c),
                     "유니크 수": df_raw[c].nunique()} for c in OHE_COLS if c in df_raw.columns]
        st.dataframe(pd.DataFrame(ohe_rows), use_container_width=True)

    with st.expander("⑤ StandardScaler (수치형)", expanded=False):
        sc_rows = [{"컬럼(원본)": c, "한국어 컬럼명": COL_KO.get(c, c),
                    "비고": "타겟 컬럼 제외"
                    if c in TARGET_COLS else "평균=0, 표준편차=1로 정규화"} for c in SCALE_COLS]
        st.dataframe(pd.DataFrame(sc_rows), use_container_width=True)

    st.divider()

    if st.button("▶ 전처리 실행", type="primary"):
        with st.spinner("전처리 중..."):
            df_proc = preprocess(df_raw)

        st.success(
            f"완료: {df_proc.shape[0]}행 × {df_proc.shape[1]}열  |  "
            f"결측치 {df_proc.isnull().sum().sum()}개"
        )
        c1, c2 = st.columns(2)
        c1.metric("원본 컬럼 수", df_raw.shape[1])
        c2.metric("전처리 후 컬럼 수", df_proc.shape[1],
                  delta=f"+{df_proc.shape[1] - df_raw.shape[1]}")

        st.subheader("전처리 결과 미리보기 (상위 10행)")
        st.dataframe(df_proc.head(10), use_container_width=True)

        st.subheader("타겟 클래스 분포 (전처리 후)")
        label_map = {
            "cgpa_category":        {0: "5.0~6.9", 1: "7.0~8.4", 2: "8.5~9.4", 3: "9.5~10.0"},
            "performance_risk_level": {0: "안정",  1: "주의 관찰",  2: "관리 필요"},
        }
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, col, title in zip(
            axes,
            ["cgpa_category", "performance_risk_level"],
            ["CGPA 구간 (인코딩 후)", "학업 관리 단계 (인코딩 후)"],
        ):
            cnt = df_proc[col].value_counts().sort_index()
            tick_labels = [label_map[col].get(int(i), str(i)) for i in cnt.index]
            bars = ax.bar(tick_labels, cnt.values,
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


# ══════════════════════════════════════════════════════════════════
# TAB 4 — 심화 분석
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧪 심화 분석")
    st.caption("복합 변수 조합 · 역설적 케이스 · 자기주도 지수 등 추가 인사이트")

    adv_section = st.radio(
        "분석 항목",
        [
            "① 복합 변수 조합",
            "② 역설적 케이스",
            "③ 학업 만족도 경로",
            "④ 스크린타임 × 생산성",
            "⑤ 외부 압박 역설",
            "⑥ SDL 자기주도 지수",
            "⑦ 진로 관심사 × 성취도",
            "⑧ 제출 시간대 분석",
            "⑨ 추가 탐구 로드맵",
        ],
        horizontal=False,
    )
    st.divider()

    # ── ① 복합 변수 조합 ───────────────────────────────────────────
    if adv_section == "① 복합 변수 조합":
        st.markdown("### ① 복합 변수 조합 분석")
        st.info("공부 시간 × 출석률, 스트레스 × 에너지 조합이 위험도에 미치는 영향을 살펴봅니다.")

        chart_sub = st.radio("차트 선택", ["공부시간 × 출석률 히트맵", "스트레스 × 에너지 복합 위험도"], horizontal=True)

        if chart_sub == "공부시간 × 출석률 히트맵":
            df_c = tdf(df_raw, ["study_hours_daily", "attendance_percentage", "performance_risk_level"])
            # 관리 필요 비율 히트맵
            cross = pd.crosstab(df_c["study_hours_daily"], df_c["attendance_percentage"])
            cross_high = pd.crosstab(
                df_c[df_c["performance_risk_level"] == "관리 필요"]["study_hours_daily"],
                df_c[df_c["performance_risk_level"] == "관리 필요"]["attendance_percentage"],
            )
            rate = (cross_high / cross * 100).fillna(0)
            rate = rate.reindex(index=study_ko, fill_value=0)
            rate = rate.reindex(columns=att_ko, fill_value=0)

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            sns.heatmap(cross.reindex(index=study_ko, fill_value=0).reindex(columns=att_ko, fill_value=0),
                        annot=True, fmt="d", cmap="Blues", ax=axes[0], linewidths=0.5)
            axes[0].set_title("공부시간 × 출석률 빈도", fontweight="bold")
            axes[0].set_xlabel("출석률")
            axes[0].set_ylabel("일일 공부 시간")
            axes[0].tick_params(axis="x", rotation=25)

            sns.heatmap(rate, annot=True, fmt=".1f", cmap="Reds", ax=axes[1],
                        linewidths=0.5, vmin=0, vmax=60)
            axes[1].set_title("공부시간 × 출석률 → 관리 필요 비율(%)", fontweight="bold")
            axes[1].set_xlabel("출석률")
            axes[1].set_ylabel("일일 공부 시간")
            axes[1].tick_params(axis="x", rotation=25)
            plt.tight_layout()
            fig2st(fig)
            st.caption("💡 공부 시간이 적고 출석률이 낮을수록 관리 필요 비율이 급격히 증가합니다.")

        else:
            df_c = tdf(df_raw, ["performance_risk_level"]).copy()
            # 스트레스 × 에너지 구간화
            df_c["스트레스_구간"] = pd.cut(df_raw["stress_level"], bins=[0, 3, 6, 10],
                                           labels=["저(1~3)", "중(4~6)", "고(7~10)"])
            df_c["에너지_구간"] = pd.cut(df_raw["energy_level"], bins=[0, 3, 6, 10],
                                         labels=["저(1~3)", "중(4~6)", "고(7~10)"])
            cross_se = pd.crosstab(df_c["스트레스_구간"], df_c["에너지_구간"])
            cross_high_se = pd.crosstab(
                df_c[df_c["performance_risk_level"] == "관리 필요"]["스트레스_구간"],
                df_c[df_c["performance_risk_level"] == "관리 필요"]["에너지_구간"],
            )
            rate_se = (cross_high_se / cross_se * 100).fillna(0)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(rate_se, annot=True, fmt=".1f", cmap="RdYlGn_r",
                        ax=ax, linewidths=0.5, vmin=0, vmax=70)
            ax.set_title("스트레스 × 에너지 구간 → 관리 필요 비율(%)", fontweight="bold")
            ax.set_xlabel("에너지 구간")
            ax.set_ylabel("스트레스 구간")
            plt.tight_layout()
            fig2st(fig)
            st.caption("💡 스트레스 높음 + 에너지 낮음 조합이 번아웃 위험 최고 구간입니다.")

    # ── ② 역설적 케이스 ───────────────────────────────────────────
    elif adv_section == "② 역설적 케이스":
        st.markdown("### ② 역설적 케이스 분석")
        st.info("공부 일관성이 '거의 안 함'임에도 관리 필요인 학생들의 프로필을 분석합니다.")

        paradox_mask = (
            (df_raw["study_consistency"] == "Rarely") &
            (df_raw["performance_risk_level"] == "High Risk")
        )
        df_paradox = df_raw[paradox_mask].copy()
        n_paradox = len(df_paradox)

        c1, c2, c3 = st.columns(3)
        c1.metric("역설 케이스 수", f"{n_paradox}명")
        c2.metric("평균 스트레스", f"{df_paradox['stress_level'].mean():.2f}")
        c3.metric("평균 에너지", f"{df_paradox['energy_level'].mean():.2f}")

        if n_paradox > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, col, title in zip(axes,
                                       ["stress_level", "energy_level", "daily_productivity"],
                                       ["스트레스 레벨", "에너지 레벨", "일일 생산성"]):
                data_all = df_raw[col].dropna()
                data_par = df_paradox[col].dropna()
                ax.hist(data_all, bins=15, alpha=0.4, color="#90CAF9", label="전체", density=True)
                ax.hist(data_par, bins=10, alpha=0.7, color="#F44336", label=f"역설 케이스(n={n_paradox})", density=True)
                ax.axvline(data_par.mean(), color="#B71C1C", linestyle="--", lw=1.5,
                           label=f"역설평균: {data_par.mean():.2f}")
                ax.axvline(data_all.mean(), color="#1565C0", linestyle=":", lw=1.5,
                           label=f"전체평균: {data_all.mean():.2f}")
                ax.set_title(title, fontweight="bold")
                ax.set_xlabel("값")
                ax.set_ylabel("밀도")
                ax.legend(fontsize=7)
            plt.suptitle("역설 케이스 vs 전체 학생 분포 비교", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            fig2st(fig)

            st.markdown("**역설 케이스 학업 만족도 분포**")
            df_par_t = tdf(df_paradox, ["academic_satisfaction"])
            sat_cnt = df_par_t["academic_satisfaction"].value_counts().reindex(sat_ko, fill_value=0)
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            bars = ax2.bar(sat_cnt.index, sat_cnt.values,
                           color=sns.color_palette("Reds", len(sat_cnt)), edgecolor="white")
            for b in bars:
                ax2.text(b.get_x() + b.get_width()/2, b.get_height()+0.3,
                         int(b.get_height()), ha="center", fontsize=9)
            ax2.set_title("역설 케이스 학업 만족도", fontweight="bold")
            ax2.set_xlabel("학업 만족도")
            ax2.set_ylabel("학생 수")
            plt.tight_layout()
            fig2st(fig2)

        st.caption("💡 '공부를 거의 안 함 + 관리 필요'인 학생은 스트레스가 높고 에너지가 낮은 번아웃 패턴을 보입니다.")

    # ── ③ 학업 만족도 경로 ───────────────────────────────────────
    elif adv_section == "③ 학업 만족도 경로":
        st.markdown("### ③ 학업 만족도 경로 분석")
        st.info("학업 만족도 수준에 따라 위험도 분포와 생산성이 어떻게 달라지는지 분석합니다.")

        df_c = tdf(df_raw, ["academic_satisfaction", "performance_risk_level"])
        cross_sat = (pd.crosstab(df_c["academic_satisfaction"], df_c["performance_risk_level"])
                     .reindex(sat_ko, fill_value=0)
                     .reindex(columns=risk_ko, fill_value=0))
        cross_pct = cross_sat.div(cross_sat.sum(axis=1), axis=0) * 100

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        cross_pct.plot(kind="bar", stacked=True,
                       color=[RISK_COLOR_KO[r] for r in risk_ko],
                       ax=axes[0], edgecolor="white", width=0.65)
        axes[0].set_title("학업 만족도 → 위험도 비율", fontweight="bold")
        axes[0].set_xlabel("학업 만족도")
        axes[0].set_ylabel("비율 (%)")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].legend(title="학업 관리 단계", bbox_to_anchor=(1, 1))

        mean_prod = df_raw.groupby(df_c["academic_satisfaction"])["daily_productivity"].mean().reindex(sat_ko)
        bars = axes[1].bar(mean_prod.index, mean_prod.values,
                           color=sns.color_palette("YlGn", len(sat_ko)), edgecolor="white")
        for b in bars:
            axes[1].text(b.get_x() + b.get_width()/2, b.get_height()+0.02,
                         f"{b.get_height():.2f}", ha="center", fontsize=9)
        axes[1].set_title("학업 만족도별 평균 생산성", fontweight="bold")
        axes[1].set_xlabel("학업 만족도")
        axes[1].set_ylabel("평균 일일 생산성")
        axes[1].set_ylim(0, mean_prod.max() * 1.2)
        axes[1].tick_params(axis="x", rotation=20)
        plt.tight_layout()
        fig2st(fig)
        st.caption("💡 '매우 불만족' 학생의 관리 필요 비율이 가장 높으며, 만족도와 생산성은 강한 양의 상관을 보입니다.")

    # ── ④ 스크린타임 × 생산성 ──────────────────────────────────────
    elif adv_section == "④ 스크린타임 × 생산성":
        st.markdown("### ④ 비학습 스크린타임 × 생산성 분석")
        st.info("비학습 스크린타임이 높을수록 생산성이 낮아지는 패턴을 확인합니다.")

        SCREEN_ORDER_EN = ["2–4 hours", "4–6 hours", "More than 6 hours"]
        screen_ko = [VAL_KO["screen_time_non_study"][v] for v in SCREEN_ORDER_EN
                     if v in df_raw["screen_time_non_study"].dropna().unique()]

        df_c = tdf(df_raw, ["screen_time_non_study", "performance_risk_level"])

        mean_prod_sc = df_raw.groupby(df_c["screen_time_non_study"])["daily_productivity"].mean().reindex(screen_ko)
        mean_stress_sc = df_raw.groupby(df_c["screen_time_non_study"])["stress_level"].mean().reindex(screen_ko)

        cross_sc = (pd.crosstab(df_c["screen_time_non_study"], df_c["performance_risk_level"])
                    .reindex(screen_ko, fill_value=0)
                    .reindex(columns=risk_ko, fill_value=0))
        cross_sc_pct = cross_sc.div(cross_sc.sum(axis=1), axis=0) * 100

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        bars = axes[0].bar(mean_prod_sc.index, mean_prod_sc.values,
                           color=["#42A5F5", "#FFA726", "#EF5350"], edgecolor="white")
        for b in bars:
            axes[0].text(b.get_x() + b.get_width()/2, b.get_height()+0.02,
                         f"{b.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")
        axes[0].set_title("스크린타임별 평균 생산성", fontweight="bold")
        axes[0].set_xlabel("비학습 스크린타임")
        axes[0].set_ylabel("평균 생산성")
        axes[0].set_ylim(0, mean_prod_sc.max() * 1.2)

        bars2 = axes[1].bar(mean_stress_sc.index, mean_stress_sc.values,
                            color=["#42A5F5", "#FFA726", "#EF5350"], edgecolor="white")
        for b in bars2:
            axes[1].text(b.get_x() + b.get_width()/2, b.get_height()+0.02,
                         f"{b.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")
        axes[1].set_title("스크린타임별 평균 스트레스", fontweight="bold")
        axes[1].set_xlabel("비학습 스크린타임")
        axes[1].set_ylabel("평균 스트레스")
        axes[1].set_ylim(0, mean_stress_sc.max() * 1.2)

        cross_sc_pct.plot(kind="bar", stacked=True,
                          color=[RISK_COLOR_KO[r] for r in risk_ko],
                          ax=axes[2], edgecolor="white", width=0.65)
        axes[2].set_title("스크린타임 → 위험도 비율", fontweight="bold")
        axes[2].set_xlabel("비학습 스크린타임")
        axes[2].set_ylabel("비율 (%)")
        axes[2].tick_params(axis="x", rotation=15)
        axes[2].legend(title="학업 관리 단계", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        fig2st(fig)
        st.caption("💡 스크린타임 2~4시간: 평균 생산성 ≈3.05, 4~6시간: 2.76, 6시간+: 2.55로 하락합니다.")

    # ── ⑤ 외부 압박 역설 ───────────────────────────────────────────
    elif adv_section == "⑤ 외부 압박 역설":
        st.markdown("### ⑤ 외부 압박 역설 분석")
        st.info("'높은 외부 압박'보다 '압박 없음'이 오히려 관리 필요 비율이 높은 역설적 패턴을 확인합니다.")

        PRESSURE_ORDER_EN = [
            "No Impact (Fully supportive environment)",
            "Low Impact (Rarely affects study)",
            "Moderate Impact (Occasional disruption)",
            "High Impact (Frequent disruption)",
        ]
        pressure_ko = [VAL_KO["external_pressure"][v] for v in PRESSURE_ORDER_EN
                       if v in df_raw["external_pressure"].dropna().unique()]

        df_c = tdf(df_raw, ["external_pressure", "performance_risk_level"])
        cross_ep = (pd.crosstab(df_c["external_pressure"], df_c["performance_risk_level"])
                    .reindex(pressure_ko, fill_value=0)
                    .reindex(columns=risk_ko, fill_value=0))
        cross_ep_pct = cross_ep.div(cross_ep.sum(axis=1), axis=0) * 100
        high_risk_pct = cross_ep_pct["관리 필요"] if "관리 필요" in cross_ep_pct.columns else cross_ep_pct.iloc[:, -1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cross_ep_pct.plot(kind="bar", stacked=True,
                          color=[RISK_COLOR_KO[r] for r in risk_ko],
                          ax=axes[0], edgecolor="white", width=0.65)
        axes[0].set_title("외부 압박 수준 → 위험도 비율", fontweight="bold")
        axes[0].set_xlabel("외부 압박")
        axes[0].set_ylabel("비율 (%)")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].legend(title="학업 관리 단계", bbox_to_anchor=(1, 1))

        colors_ep = ["#66BB6A" if v == high_risk_pct.min() else "#EF5350"
                     if v == high_risk_pct.max() else "#90CAF9" for v in high_risk_pct.values]
        bars = axes[1].bar(high_risk_pct.index, high_risk_pct.values, color=colors_ep, edgecolor="white")
        for b in bars:
            axes[1].text(b.get_x() + b.get_width()/2, b.get_height()+0.3,
                         f"{b.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
        axes[1].set_title("외부 압박별 관리 필요 비율", fontweight="bold")
        axes[1].set_xlabel("외부 압박")
        axes[1].set_ylabel("관리 필요 비율 (%)")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].set_ylim(0, high_risk_pct.max() * 1.25)
        plt.tight_layout()
        fig2st(fig)
        st.caption("💡 '압박 없음(완전 지지 환경)' 그룹이 오히려 관리 필요 비율이 높습니다 — 압박 부재 = 동기 결여 가설.")

    # ── ⑥ SDL 자기주도 지수 ────────────────────────────────────────
    elif adv_section == "⑥ SDL 자기주도 지수":
        st.markdown("### ⑥ SDL (자기주도 학습) 지수 분석")
        st.info("study_consistency · revision_frequency · tasks_on_time · assignments_on_time를 합산해 SDL 지수를 산출합니다.")

        SDL_MAPS = {
            "study_consistency":  {"Rarely": 0, "Sometimes": 1, "Mostly consistent": 2},
            "revision_frequency": {"Never": 0, "Rarely": 1, "Few times a week": 2, "Daily": 3},
            "tasks_on_time":      {"Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3},
            "assignments_on_time":{"Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3},
        }
        df_sdl = df_raw.copy()
        for col, mapping in SDL_MAPS.items():
            if col in df_sdl.columns:
                df_sdl[col + "_enc"] = df_sdl[col].map(mapping)

        enc_cols = [c + "_enc" for c in SDL_MAPS if c in df_raw.columns]
        df_sdl["SDL_지수"] = df_sdl[enc_cols].sum(axis=1)
        sdl_max = sum(max(v.values()) for v in SDL_MAPS.values())

        df_sdl_t = tdf(df_sdl, ["performance_risk_level", "cgpa_category"])

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].hist(df_sdl["SDL_지수"].dropna(), bins=range(0, sdl_max + 2),
                     color="#42A5F5", edgecolor="white", alpha=0.85)
        axes[0].set_title(f"SDL 지수 분포 (최대 {sdl_max}점)", fontweight="bold")
        axes[0].set_xlabel("SDL 지수")
        axes[0].set_ylabel("학생 수")

        sns.boxplot(data=df_sdl_t, x="performance_risk_level", y="SDL_지수",
                    order=risk_ko, palette=RISK_COLOR_KO, ax=axes[1], width=0.5)
        axes[1].set_title("위험도별 SDL 지수 분포", fontweight="bold")
        axes[1].set_xlabel("학업 관리 단계")
        axes[1].set_ylabel("SDL 지수")

        mean_sdl_cgpa = df_sdl.groupby(df_sdl_t["cgpa_category"])["SDL_지수"].mean().reindex(cgpa_ko)
        bars = axes[2].bar(mean_sdl_cgpa.index, mean_sdl_cgpa.values,
                           color=sns.color_palette("Blues_d", len(cgpa_ko)), edgecolor="white")
        for b in bars:
            axes[2].text(b.get_x() + b.get_width()/2, b.get_height()+0.05,
                         f"{b.get_height():.1f}", ha="center", fontsize=9)
        axes[2].set_title("CGPA 구간별 평균 SDL 지수", fontweight="bold")
        axes[2].set_xlabel("CGPA 구간")
        axes[2].set_ylabel("평균 SDL 지수")
        axes[2].tick_params(axis="x", rotation=15)
        axes[2].set_ylim(0, mean_sdl_cgpa.max() * 1.2)
        plt.tight_layout()
        fig2st(fig)

        sdl_risk_mean = df_sdl.groupby(df_sdl_t["performance_risk_level"])["SDL_지수"].mean().reindex(risk_ko)
        st.markdown("**위험도별 SDL 평균**")
        sdl_table = pd.DataFrame({
            "학업 관리 단계": sdl_risk_mean.index,
            "SDL 평균 (만점 {})".format(sdl_max): sdl_risk_mean.values.round(2),
        })
        st.dataframe(sdl_table, use_container_width=True)
        st.caption("💡 SDL 지수가 낮을수록 관리 필요 비율이 높으며, SDL은 단일 변수보다 강한 예측력을 가집니다.")

    # ── ⑦ 진로 관심사 × 성취도 ────────────────────────────────────
    elif adv_section == "⑦ 진로 관심사 × 성취도":
        st.markdown("### ⑦ 진로 관심사 × 학업 성취도 분석")
        st.info("진로 관심사에 따라 고CGPA(8.5 이상) 비율과 위험도 분포가 어떻게 다른지 비교합니다.")

        HIGH_CGPA = {"8.5 – 9.4", "9.5 – 10.0"}
        df_ci = df_raw.copy()
        df_ci["고CGPA"] = df_ci["cgpa_category"].isin(HIGH_CGPA).astype(int)
        df_ci_t = tdf(df_ci, ["career_interest", "performance_risk_level"])

        career_order_ko = (df_ci_t["career_interest"].value_counts().index.tolist())

        high_cgpa_rate = df_ci.groupby(df_ci_t["career_interest"])["고CGPA"].mean() * 100
        high_cgpa_rate = high_cgpa_rate.reindex(career_order_ko)

        cross_ci = (pd.crosstab(df_ci_t["career_interest"], df_ci_t["performance_risk_level"])
                    .reindex(career_order_ko, fill_value=0)
                    .reindex(columns=risk_ko, fill_value=0))
        cross_ci_pct = cross_ci.div(cross_ci.sum(axis=1), axis=0) * 100

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        colors_ci = ["#FFA726" if v >= high_cgpa_rate.median() else "#90CAF9"
                     for v in high_cgpa_rate.values]
        bars = axes[0].barh(high_cgpa_rate.index[::-1], high_cgpa_rate.values[::-1],
                            color=colors_ci[::-1], edgecolor="white")
        for b in bars:
            axes[0].text(b.get_width() + 0.3, b.get_y() + b.get_height()/2,
                         f"{b.get_width():.1f}%", va="center", fontsize=9)
        axes[0].set_title("진로 관심사별 고CGPA(8.5+) 비율", fontweight="bold")
        axes[0].set_xlabel("고CGPA 비율 (%)")
        axes[0].set_xlim(0, high_cgpa_rate.max() * 1.25)

        cross_ci_pct.plot(kind="barh", stacked=True,
                          color=[RISK_COLOR_KO[r] for r in risk_ko],
                          ax=axes[1], edgecolor="white")
        axes[1].set_title("진로 관심사별 위험도 비율", fontweight="bold")
        axes[1].set_xlabel("비율 (%)")
        axes[1].set_ylabel("")
        axes[1].legend(title="학업 관리 단계", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        fig2st(fig)
        st.caption("💡 소프트웨어 개발자·데이터 분석가 지망생에서 고CGPA 비율이 상대적으로 높게 나타납니다.")

    # ── ⑧ 제출 시간대 분석 ─────────────────────────────────────────
    elif adv_section == "⑧ 제출 시간대 분석":
        st.markdown("### ⑧ 타임스탬프 제출 시간대 분석")
        st.info("설문 제출 시간(timestamp)을 시간대로 분류하여 수면 패턴 및 위험도와의 관계를 탐색합니다.")

        if "timestamp" not in df_raw.columns:
            st.warning("timestamp 컬럼이 데이터에 없습니다.")
        else:
            df_ts = df_raw.copy()
            try:
                df_ts["dt"] = pd.to_datetime(df_ts["timestamp"], errors="coerce")
                df_ts["시간"] = df_ts["dt"].dt.hour
                df_ts_valid = df_ts.dropna(subset=["시간"])

                def hour_to_period(h):
                    if 5 <= h < 12:
                        return "오전(5~12시)"
                    elif 12 <= h < 18:
                        return "오후(12~18시)"
                    elif 18 <= h < 23:
                        return "저녁(18~23시)"
                    else:
                        return "심야(23~5시)"

                df_ts_valid = df_ts_valid.copy()
                df_ts_valid["시간대"] = df_ts_valid["시간"].apply(hour_to_period)
                period_order = ["오전(5~12시)", "오후(12~18시)", "저녁(18~23시)", "심야(23~5시)"]
                df_ts_t = tdf(df_ts_valid, ["performance_risk_level", "sleep_hours"])

                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                hour_cnt = df_ts_valid["시간"].value_counts().sort_index()
                axes[0].bar(hour_cnt.index, hour_cnt.values,
                            color=sns.color_palette("twilight", 24), edgecolor="white")
                axes[0].set_title("시간대별 제출 수 (24h)", fontweight="bold")
                axes[0].set_xlabel("시각 (0~23시)")
                axes[0].set_ylabel("제출 수")
                axes[0].set_xticks(range(0, 24, 3))

                period_cnt = df_ts_valid["시간대"].value_counts().reindex(period_order, fill_value=0)
                axes[1].pie(period_cnt.values, labels=period_cnt.index, autopct="%1.1f%%",
                            colors=["#64B5F6", "#FFA726", "#EF5350", "#7E57C2"],
                            startangle=140, wedgeprops={"edgecolor": "white"})
                axes[1].set_title("시간대 비율", fontweight="bold")

                cross_period = (pd.crosstab(df_ts_valid["시간대"], df_ts_t["performance_risk_level"])
                                .reindex(period_order, fill_value=0)
                                .reindex(columns=risk_ko, fill_value=0))
                cross_period_pct = cross_period.div(cross_period.sum(axis=1), axis=0) * 100
                cross_period_pct.plot(kind="bar", stacked=True,
                                      color=[RISK_COLOR_KO[r] for r in risk_ko],
                                      ax=axes[2], edgecolor="white")
                axes[2].set_title("제출 시간대 → 위험도 비율", fontweight="bold")
                axes[2].set_xlabel("시간대")
                axes[2].set_ylabel("비율 (%)")
                axes[2].tick_params(axis="x", rotation=20)
                axes[2].legend(title="학업 관리 단계", bbox_to_anchor=(1, 1))
                plt.tight_layout()
                fig2st(fig)
                st.caption("💡 심야 제출 학생은 수면 부족 및 관리 필요 비율이 상대적으로 높은 경향이 있습니다.")
            except Exception as e:
                st.error(f"timestamp 파싱 오류: {e}")
                st.write("timestamp 샘플:", df_raw["timestamp"].dropna().head(5).tolist())

    # ── ⑨ 추가 탐구 로드맵 ─────────────────────────────────────────
    elif adv_section == "⑨ 추가 탐구 로드맵":
        st.markdown("### ⑨ 추가 탐구 우선순위 로드맵")
        st.info("데이터에서 도출된 추가 연구 방향과 예상 난이도를 정리합니다.")

        roadmap_data = {
            "분석 항목": [
                "ML 예측 모델 구축 (Random Forest / XGBoost)",
                "SMOTE로 클래스 불균형 보정 후 재학습",
                "SDL 지수 → 성취도 회귀 분석",
                "번아웃 클러스터링 (K-Means / DBSCAN)",
                "LLM 맞춤형 학습 상담 리포트 생성",
                "특성 중요도 (SHAP 값 시각화)",
                "Streamlit 실시간 예측 인터페이스",
                "시계열 종단 연구 설계 (추가 데이터 수집)",
            ],
            "우선순위": ["★★★", "★★★", "★★☆", "★★☆", "★★★", "★★☆", "★★★", "★☆☆"],
            "예상 난이도": ["중", "중", "하", "중", "고", "중", "중", "고"],
            "기대 인사이트": [
                "위험 학생 조기 식별 모델",
                "안정(15%) 재현율 향상",
                "SDL이 CGPA에 미치는 정량적 영향",
                "학생 유형 군집화",
                "개인 맞춤 학습 전략 제안",
                "어떤 변수가 예측에 핵심인지",
                "실용적 학생 지원 서비스",
                "성취도 변화 추적",
            ],
        }
        roadmap_df = pd.DataFrame(roadmap_data)
        st.dataframe(roadmap_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 핵심 발견 요약")

        insights = [
            ("공부 일관성", "가장 강력한 단일 예측 변수 — '거의 안 함' 그룹의 관리 필요 비율 ≈74%"),
            ("SDL 복합 지수", "study_consistency + revision + tasks/assignments_on_time 합산 → 예측력 향상"),
            ("번아웃 패턴", "스트레스 高 + 에너지 低 + 공부 일관성 低 → 관리 필요 최고 구간"),
            ("외부 압박 역설", "'압박 없음' 그룹의 관리 필요 비율이 '높은 압박' 그룹보다 높음"),
            ("클래스 불균형", "안정 15%(176명) — 모델 학습 시 SMOTE 또는 class_weight 필수"),
            ("스크린타임", "6시간 초과 그룹의 평균 생산성 ≈2.55 (2~4시간 그룹 3.05 대비 16% 저하)"),
        ]
        for title, desc in insights:
            st.markdown(f"- **{title}**: {desc}")

        st.markdown("---")
        st.markdown("#### 데이터 한계 및 주의사항")
        st.warning(
            "• 현재 데이터셋은 단면(Cross-sectional) 조사로, 인과관계가 아닌 상관관계만 추론 가능합니다.\n"
            "• 자기 보고 편향(Self-report Bias)이 있을 수 있으며, 학교/전공별 맥락 차이를 고려해야 합니다.\n"
            "• 안정 클래스(15%)의 심각한 불균형으로 단순 정확도(Accuracy)는 신뢰할 수 없습니다.\n"
            "• 추후 종단 연구(Longitudinal Study)나 개입 실험(Intervention Study)으로 검증이 필요합니다."
        )


# ══════════════════════════════════════════════════════════════════
# TAB 5 — 필터 분석
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🎛️ 인터랙티브 필터 분석")
    st.caption("슬라이더·다중 선택으로 부분집합을 설정하면 상관관계와 분포가 실시간 갱신됩니다.")

    # ── 전체 번역 df (필터 적용용) ────────────────────────────────
    _df_t_full = tdf(df_raw, list(VAL_KO.keys()))

    # ── 수치형 실제 범위 ─────────────────────────────────────────
    NUM_FILTER_COLS = ["stress_level", "energy_level", "daily_productivity", "routine_rating", "age"]
    num_ranges = {
        c: (int(df_raw[c].min()), int(df_raw[c].max()))
        for c in NUM_FILTER_COLS if c in df_raw.columns
    }

    # ── 범주형 선택 옵션 (실제 데이터에 존재하는 값만) ────────────
    def cat_opts(col, order_ko=None):
        vals = _df_t_full[col].dropna().unique().tolist()
        if order_ko:
            return [v for v in order_ko if v in vals]
        return sorted(vals)

    # ── 레이아웃: 필터 패널 | 결과 영역 ─────────────────────────
    f_col, r_col = st.columns([1, 2.8], gap="large")

    with f_col:
        st.markdown("### 필터 설정")

        st.markdown("**수치형 범위**")
        s_stress   = st.slider("스트레스 레벨",    *num_ranges.get("stress_level",   (1,10)), (num_ranges.get("stress_level",   (1,10))[0], num_ranges.get("stress_level",   (1,10))[1]))
        s_energy   = st.slider("에너지 레벨",      *num_ranges.get("energy_level",   (1,10)), (num_ranges.get("energy_level",   (1,10))[0], num_ranges.get("energy_level",   (1,10))[1]))
        s_prod     = st.slider("일일 생산성",      *num_ranges.get("daily_productivity",(1,10)), (num_ranges.get("daily_productivity",(1,10))[0], num_ranges.get("daily_productivity",(1,10))[1]))
        s_routine  = st.slider("루틴 평가",        *num_ranges.get("routine_rating", (1,10)), (num_ranges.get("routine_rating", (1,10))[0], num_ranges.get("routine_rating", (1,10))[1]))
        age_lo, age_hi = num_ranges.get("age", (18, 30))
        s_age      = st.slider("나이",             age_lo, age_hi, (age_lo, age_hi))

        st.markdown("**범주형 선택**")
        sel_risk   = st.multiselect("학업 관리 단계",  cat_opts("performance_risk_level", risk_ko),  default=cat_opts("performance_risk_level", risk_ko))
        sel_study  = st.multiselect("일일 공부 시간",  cat_opts("study_hours_daily",  study_ko),  default=cat_opts("study_hours_daily",  study_ko))
        sel_att    = st.multiselect("출석률",          cat_opts("attendance_percentage", att_ko),  default=cat_opts("attendance_percentage", att_ko))
        sel_sleep  = st.multiselect("수면 시간",       cat_opts("sleep_hours", sleep_ko),          default=cat_opts("sleep_hours", sleep_ko))
        sel_gender = st.multiselect("성별",            cat_opts("gender"),                          default=cat_opts("gender"))

        st.markdown("---")
        if st.button("🔄 필터 초기화", use_container_width=True):
            st.rerun()

    # ── 필터 적용 ─────────────────────────────────────────────────
    mask5 = pd.Series(True, index=df_raw.index)

    for col, rng in [("stress_level", s_stress), ("energy_level", s_energy),
                     ("daily_productivity", s_prod), ("routine_rating", s_routine),
                     ("age", s_age)]:
        if col in df_raw.columns:
            notna = df_raw[col].notna()
            mask5 &= (~notna) | df_raw[col].between(*rng)

    for col, sel in [("performance_risk_level", sel_risk), ("study_hours_daily", sel_study),
                     ("attendance_percentage", sel_att), ("sleep_hours", sel_sleep),
                     ("gender", sel_gender)]:
        if col in _df_t_full.columns and sel:
            mask5 &= _df_t_full[col].isin(sel)

    df5     = df_raw[mask5].copy()
    df5_t   = _df_t_full[mask5].copy()
    n_raw   = len(df_raw)
    n_filt  = len(df5)

    with r_col:
        # ── 요약 지표 ──────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전체 학생", n_raw)
        m2.metric("필터 후 학생", n_filt, delta=f"{n_filt - n_raw:+d}")
        if n_filt > 0:
            risk_pct = df5_t["performance_risk_level"].value_counts(normalize=True) * 100
            m3.metric("관리 필요", f"{risk_pct.get('관리 필요', 0):.1f}%")
            m4.metric("안정",     f"{risk_pct.get('안정', 0):.1f}%")
        else:
            m3.metric("관리 필요", "—")
            m4.metric("안정", "—")

        st.divider()

        if n_filt == 0:
            st.warning("조건에 맞는 학생이 없습니다. 필터를 조정해 주세요.")
        else:
            chart5 = st.radio(
                "표시할 분석",
                ["학업 관리 단계 분포", "수치형 상관관계 히트맵",
                 "스트레스 × 에너지 산점도", "변수별 평균 비교"],
                horizontal=True,
            )
            st.divider()

            # ── 학업 관리 단계 분포 ───────────────────────────
            if chart5 == "학업 관리 단계 분포":
                risk_cnt_filt  = df5_t["performance_risk_level"].value_counts().reindex(risk_ko, fill_value=0)
                risk_cnt_total = _df_t_full["performance_risk_level"].value_counts().reindex(risk_ko, fill_value=0)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                # 필터 후 파이
                axes[0].pie(
                    risk_cnt_filt.values,
                    labels=risk_cnt_filt.index,
                    autopct="%1.1f%%",
                    colors=[RISK_COLOR_KO[r] for r in risk_ko],
                    startangle=140,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                    textprops={"fontsize": 11},
                )
                axes[0].set_title(f"필터 후 분포 (n={n_filt})", fontweight="bold")

                # 전체 vs 필터 비율 막대 비교
                df_cmp = pd.DataFrame({
                    "전체": (risk_cnt_total / risk_cnt_total.sum() * 100),
                    "필터 후": (risk_cnt_filt / risk_cnt_filt.sum() * 100) if risk_cnt_filt.sum() > 0 else risk_cnt_filt * 0,
                })
                df_cmp.plot(kind="bar", ax=axes[1],
                            color=["#90CAF9", "#F48FB1"], edgecolor="white", width=0.6)
                axes[1].set_title("전체 vs 필터 후 비율 비교", fontweight="bold")
                axes[1].set_xlabel("학업 관리 단계")
                axes[1].set_ylabel("비율 (%)")
                axes[1].tick_params(axis="x", rotation=15)
                axes[1].legend()
                plt.tight_layout()
                fig2st(fig)

            # ── 수치형 상관관계 히트맵 ────────────────────────
            elif chart5 == "수치형 상관관계 히트맵":
                num_cols5 = df5.select_dtypes("number").columns.tolist()
                if len(num_cols5) >= 2:
                    corr5 = df5[num_cols5].corr()
                    corrA = df_raw[num_cols5].corr()  # 전체 상관계수 (비교용)
                    corr5.columns = [ko(c) for c in corr5.columns]
                    corr5.index   = [ko(c) for c in corr5.index]
                    mask_tri = np.triu(np.ones_like(corr5, dtype=bool))

                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    for ax, corr_data, title in [
                        (axes[0], corr5, f"필터 후 상관관계 (n={n_filt})"),
                        (axes[1], corrA.rename(columns=ko, index=ko), f"전체 상관관계 (n={n_raw})"),
                    ]:
                        corrA_indexed = corr_data.rename(columns=lambda c: COL_KO.get(c, c),
                                                          index=lambda c: COL_KO.get(c, c))
                        sns.heatmap(corr_data, mask=mask_tri, annot=True, fmt=".2f",
                                    cmap="coolwarm", center=0, vmin=-1, vmax=1,
                                    linewidths=0.5, annot_kws={"size": 9}, ax=ax)
                        ax.set_title(title, fontweight="bold")
                    plt.tight_layout()
                    fig2st(fig)

                    # 상관계수 변화량 표
                    corrA2 = df_raw[df5.select_dtypes("number").columns].corr()
                    diff = (corr5.values - np.tril(corrA2.rename(columns=ko, index=ko).reindex(index=corr5.index, columns=corr5.columns).values))
                    st.caption("▶ 필터 전후 수치형 상관계수 상위 변화 쌍")
                    pairs5 = (corr5.where(~mask_tri).stack().reset_index()
                              .rename(columns={"level_0": "변수1", "level_1": "변수2", 0: "필터후 상관계수"}))
                    pairs5["전체 상관계수"] = (corrA.rename(columns=ko, index=ko)
                                                .reindex(index=corr5.index, columns=corr5.columns)
                                                .where(~mask_tri).stack().values)
                    pairs5["변화량"] = (pairs5["필터후 상관계수"] - pairs5["전체 상관계수"]).round(3)
                    pairs5 = pairs5.assign(abs변화=pairs5["변화량"].abs()).sort_values("abs변화", ascending=False).drop("abs변화", axis=1).head(6).reset_index(drop=True)
                    pairs5["필터후 상관계수"] = pairs5["필터후 상관계수"].round(3)
                    pairs5["전체 상관계수"]   = pairs5["전체 상관계수"].round(3)
                    st.dataframe(pairs5, use_container_width=True)
                else:
                    st.info("수치형 컬럼이 2개 이상 필요합니다.")

            # ── 스트레스 × 에너지 산점도 ─────────────────────
            elif chart5 == "스트레스 × 에너지 산점도":
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                for ax, dft, label, n in [
                    (axes[0], df5_t,        f"필터 후 (n={n_filt})", n_filt),
                    (axes[1], _df_t_full,   f"전체 (n={n_raw})",     n_raw),
                ]:
                    for risk in risk_ko:
                        g = dft[dft["performance_risk_level"] == risk]
                        ax.scatter(
                            df_raw.loc[g.index, "stress_level"],
                            df_raw.loc[g.index, "energy_level"],
                            c=RISK_COLOR_KO[risk], label=risk,
                            alpha=0.5, s=28, edgecolors="white", linewidths=0.3,
                        )
                    ax.set_title(f"스트레스 × 에너지 — {label}", fontweight="bold")
                    ax.set_xlabel("스트레스 레벨")
                    ax.set_ylabel("에너지 레벨")
                    ax.legend(title="학업 관리 단계", fontsize=8)
                plt.tight_layout()
                fig2st(fig)

            # ── 변수별 평균 비교 ──────────────────────────────
            else:
                NUM5 = ["stress_level", "energy_level", "daily_productivity", "routine_rating"]
                num5_exist = [c for c in NUM5 if c in df5.columns and c in df_raw.columns]
                mean_filt  = df5[num5_exist].mean()
                mean_total = df_raw[num5_exist].mean()
                mean_df    = pd.DataFrame({
                    "전체 평균":   mean_total,
                    "필터 후 평균": mean_filt,
                })
                mean_df.index = [ko(c) for c in mean_df.index]

                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(mean_df))
                w = 0.35
                bars1 = ax.bar(x - w/2, mean_df["전체 평균"],   w, label="전체",   color="#90CAF9", edgecolor="white")
                bars2 = ax.bar(x + w/2, mean_df["필터 후 평균"], w, label="필터 후", color="#F48FB1", edgecolor="white")
                for b in list(bars1) + list(bars2):
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,
                            f"{b.get_height():.2f}", ha="center", fontsize=8, fontweight="bold")
                ax.set_title(f"수치형 변수 평균 — 전체 vs 필터 후 (n={n_filt})", fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(mean_df.index, rotation=10)
                ax.set_ylabel("평균값")
                ax.legend()
                ax.set_ylim(0, max(mean_df.max().max() * 1.25, 1))
                plt.tight_layout()
                fig2st(fig)

                # 수치 표
                mean_df["변화량"] = (mean_df["필터 후 평균"] - mean_df["전체 평균"]).round(2)
                mean_df["변화(%)"] = ((mean_df["필터 후 평균"] - mean_df["전체 평균"]) / mean_df["전체 평균"] * 100).round(1).astype(str) + "%"
                st.dataframe(mean_df.round(3), use_container_width=True)

            # ── 필터된 원시 데이터 보기 ───────────────────────
            st.divider()
            with st.expander(f"📄 필터된 원시 데이터 보기 ({n_filt}행)", expanded=False):
                show_cols = ["performance_risk_level", "cgpa_category",
                             "stress_level", "energy_level", "daily_productivity",
                             "study_hours_daily", "attendance_percentage", "sleep_hours", "gender"]
                show_cols = [c for c in show_cols if c in df5.columns]
                st.dataframe(
                    tdf(df5, show_cols).rename(columns=COL_KO).head(200),
                    use_container_width=True,
                )
                if n_filt <= 1000:
                    csv5 = tdf(df5, list(VAL_KO.keys())).rename(columns=COL_KO)\
                               .to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button("⬇ 필터 데이터 CSV 다운로드", csv5,
                                       "filtered_students.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════
# TAB 6 — 발표 슬라이드
# ══════════════════════════════════════════════════════════════════
with tab6:
    import base64 as _b64, io as _bio


    # ── base64 헬퍼 ─────────────────────────────────────────────
    def _b64img(fig, dpi=130):
        buf = _bio.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        return "data:image/png;base64," + _b64.b64encode(buf.read()).decode()

    # ── 슬라이드 3용 차트 A: 파이차트 ─────────────────────────
    _ds = tdf(df_raw, ["performance_risk_level"])
    _rc = _ds["performance_risk_level"].value_counts().reindex(risk_ko, fill_value=0)
    fA, aA = plt.subplots(figsize=(4.8, 3.8), facecolor="white")
    aA.pie(_rc.values, labels=_rc.index, autopct="%1.1f%%",
           colors=[RISK_COLOR_KO[r] for r in risk_ko], startangle=140,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5},
           textprops={"fontsize": 12})
    aA.set_title("학업 관리 단계 분포  (n=1,200)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    imgA = _b64img(fA)
    plt.close(fA)

    # ── 슬라이드 3용 차트 B: 공부 일관성 × 관리 필요 비율 ──────
    _CONS_EN = ["Rarely", "Sometimes", "Mostly consistent"]
    _cons_ko = [VAL_KO["study_consistency"][v] for v in _CONS_EN
                if v in df_raw["study_consistency"].dropna().unique()]
    _dcs = tdf(df_raw, ["study_consistency", "performance_risk_level"])
    _crt = (pd.crosstab(_dcs["study_consistency"], _dcs["performance_risk_level"])
            .reindex(_cons_ko, fill_value=0).reindex(columns=risk_ko, fill_value=0))
    _hp  = (_crt["관리 필요"] / _crt.sum(axis=1) * 100).reindex(_cons_ko)
    fB, aB = plt.subplots(figsize=(4.8, 3.8), facecolor="white")
    _brs = aB.bar(_hp.index, _hp.values,
                  color=["#EF5350", "#FF8A65", "#66BB6A"][:len(_hp)],
                  edgecolor="white", width=0.5)
    for b in _brs:
        aB.text(b.get_x()+b.get_width()/2, b.get_height()+0.8,
                f"{b.get_height():.0f}%", ha="center", fontsize=13, fontweight="bold")
    aB.set_title("공부 일관성 → 관리 필요 비율", fontsize=12, fontweight="bold")
    aB.set_xlabel("공부 일관성", fontsize=10); aB.set_ylabel("관리 필요 비율 (%)", fontsize=10)
    aB.set_ylim(0, 90)
    plt.tight_layout()
    imgB = _b64img(fB)
    plt.close(fB)

    # ── 슬라이드 5용 차트 C: SDL 박스플롯 ─────────────────────
    _SDL_M = {
        "study_consistency":   {"Rarely": 0, "Sometimes": 1, "Mostly consistent": 2},
        "revision_frequency":  {"Never": 0, "Rarely": 1, "Few times a week": 2, "Daily": 3},
        "tasks_on_time":       {"Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3},
        "assignments_on_time": {"Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3},
    }
    _ds2 = df_raw.copy()
    for _c, _m in _SDL_M.items():
        if _c in _ds2.columns:
            _ds2[_c+"_e"] = _ds2[_c].map(_m)
    _ds2["SDL"] = _ds2[[_c+"_e" for _c in _SDL_M if _c in df_raw.columns]].sum(axis=1)
    _ds2t = tdf(_ds2, ["performance_risk_level"])
    fC, aC = plt.subplots(figsize=(5, 3.8), facecolor="white")
    sns.boxplot(data=_ds2t, x="performance_risk_level", y="SDL",
                order=risk_ko, palette=RISK_COLOR_KO, ax=aC, width=0.45)
    aC.set_title("학업 관리 단계별 SDL 지수", fontsize=12, fontweight="bold")
    aC.set_xlabel("학업 관리 단계", fontsize=10); aC.set_ylabel("SDL 지수 (0~11)", fontsize=10)
    plt.tight_layout()
    imgC = _b64img(fC)
    plt.close(fC)

    # ── CSS ─────────────────────────────────────────────────────
    st.markdown("""
<style>
.deck { font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; color: #212121 !important; }

.slide {
    background: #ffffff;
    border: 1px solid #dde3ea;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,.10);
    padding: 48px 64px;
    margin: 0 0 36px 0;
    min-height: 490px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    overflow: hidden;
}
.slide::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 6px; height: 100%;
    background: #1565C0;
}

.snum {
    position: absolute; top: 14px; right: 22px;
    font-size: 11px; color: #78909c !important; letter-spacing: 1px;
}
.stitle {
    font-size: 2.0rem; font-weight: 700;
    color: #1a237e; margin: 0 0 10px 0; line-height: 1.25;
}
.ssub  { font-size: 1.05rem; color: #37474f !important; margin: 0 0 20px 0; }
.slabel {
    font-size: 0.78rem; font-weight: 700; color: #1565C0;
    text-transform: uppercase; letter-spacing: 1.2px; margin: 0 0 6px 0;
}

.two-col { display: flex; gap: 28px; align-items: flex-start; }
.two-col > div { flex: 1; }

.kpi-row { display: flex; gap: 20px; margin: 20px 0; }
.kpi {
    flex: 1; background: #E3F2FD; border-radius: 8px;
    padding: 14px 20px; text-align: center;
}
.kpi-num { font-size: 2.4rem; font-weight: 700; color: #1565C0; line-height: 1; }
.kpi-lbl { font-size: 0.82rem; color: #546e7a; margin-top: 4px; }

.tag {
    display: inline-block; border-radius: 20px;
    padding: 3px 13px; font-size: 0.85rem; font-weight: 600; margin: 3px;
}
.tg  { background: #E3F2FD; color: #1565C0; }
.tgr { background: #FFEBEE; color: #C62828; }
.tgy { background: #FFF8E1; color: #E65100; }
.tgg { background: #E8F5E9; color: #2E7D32; }

.insight {
    background: #FFF8E1;
    border-left: 4px solid #FFC107;
    padding: 10px 18px; border-radius: 0 6px 6px 0;
    font-size: 1.0rem; font-weight: 600; margin-top: 16px;
}

.demo-box {
    text-align: center;
    padding: 28px;
    background: linear-gradient(135deg, #1a237e 0%, #1976D2 100%);
    border-radius: 12px; color: white; margin: 12px 0;
}
.demo-main { font-size: 2.8rem; font-weight: 700; }
.demo-url  { font-size: 1.1rem; opacity: .80; margin-top: 6px; letter-spacing: .5px; }

.steps { list-style: none; padding: 0; margin: 18px 0 0 0; }
.steps li {
    display: flex; align-items: center; gap: 12px;
    padding: 7px 0; font-size: 0.95rem; color: #212121 !important;
    border-bottom: 1px solid #dde3ea;
}
.snum-badge {
    background: #1565C0; color: white; border-radius: 50%;
    width: 26px; height: 26px; display: flex;
    align-items: center; justify-content: center;
    font-size: 0.82rem; font-weight: 700; flex-shrink: 0;
}

.formula {
    background: #F3F8FF; border: 1px solid #BBDEFB;
    border-radius: 8px; padding: 16px 22px;
    font-size: 0.95rem; font-weight: 500; color: #1a237e;
    margin: 14px 0; line-height: 1.7;
}

.concl-row { display: flex; gap: 18px; margin-top: 22px; }
.ccard {
    flex: 1; padding: 18px 20px; border-radius: 8px; border-top: 4px solid;
}
.ccard h4 { margin: 0 0 7px 0; font-size: 0.82rem; text-transform: uppercase; letter-spacing: .6px; }
.ccard p  { margin: 0; font-size: 0.92rem; color: #212121 !important; line-height: 1.5; }
.ccard.cg { border-color: #4CAF50; background: #F1F8E9; }
.ccard.cg h4 { color: #2E7D32; }
.ccard.co { border-color: #FF9800; background: #FFF3E0; }
.ccard.co h4 { color: #E65100; }
.ccard.cb { border-color: #2196F3; background: #E3F2FD; }
.ccard.cb h4 { color: #1565C0; }

.title-slide { background: linear-gradient(135deg, #E8EAF6 0%, #FFFFFF 100%); }
.title-slide::before { background: #283593; }

/* ── 인쇄 ── */
@media print {
    @page { size: A4 landscape; margin: 0; }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-baseweb="tab-list"],
    [data-testid="stSidebar"],
    .stAlert, footer, header { display: none !important; }

    .deck { display: block !important; }

    .slide {
        page-break-after: always;
        page-break-inside: avoid;
        box-shadow: none !important;
        border: none !important;
        border-radius: 0 !important;
        margin: 0 !important;
        padding: 36px 60px !important;
        min-height: 0 !important;
        width: 100% !important;
    }
    .slide:last-child { page-break-after: avoid; }
}
</style>
""", unsafe_allow_html=True)

    # ── 슬라이드 HTML ────────────────────────────────────────────
    st.markdown(f"""
<div class="deck">

<!-- ★ SLIDE 1: 제목 -->
<div class="slide title-slide">
  <span class="snum">01 / 06</span>
  <div style="margin-bottom:8px">
    <span class="tag tg">빅데이터 프로젝트</span>
    <span class="tag tg">2026-06-19</span>
  </div>
  <div class="stitle">학생 성취도 분석 대시보드</div>
  <div class="ssub">행동 데이터로 학습 지원 필요 학생을 조기 식별하는 인터랙티브 앱</div>
  <div style="margin-top:32px; font-size:1.1rem; color:#263238 !important;">
    <strong>정명진</strong> &nbsp;/&nbsp; 20242530
  </div>
</div>

<!-- ★ SLIDE 2: 문제 정의 & 데이터 -->
<div class="slide">
  <span class="snum">02 / 06</span>
  <div class="slabel">Problem &amp; Data</div>
  <div class="stitle">무엇을, 왜 분석하는가</div>
  <div class="two-col" style="margin-top:20px;">
    <div>
      <p style="font-size:1.05rem; color:#212121 !important; line-height:1.7; margin:0 0 16px 0;">
        학업 성취도는 공부 시간만이 아니라<br>
        수면·스트레스·습관 등 복합 요인이 결정한다.
      </p>
      <div style="margin-bottom:8px"><span class="slabel">예측 타겟</span></div>
      <span class="tag tgg">안정</span>
      <span class="tag tgy">주의 관찰</span>
      <span class="tag tgr">관리 필요</span>
      <span class="tag tg" style="margin-top:8px; display:inline-block;">CGPA 구간 (4단계)</span>
    </div>
    <div>
      <div class="kpi-row">
        <div class="kpi">
          <div class="kpi-num">1,200</div>
          <div class="kpi-lbl">학생 수 (행)</div>
        </div>
        <div class="kpi">
          <div class="kpi-num">36</div>
          <div class="kpi-lbl">변수 수 (열)</div>
        </div>
      </div>
      <div class="kpi-row">
        <div class="kpi">
          <div class="kpi-num">13</div>
          <div class="kpi-lbl">결측 컬럼</div>
        </div>
        <div class="kpi">
          <div class="kpi-num">~80</div>
          <div class="kpi-lbl">전처리 후 열</div>
        </div>
      </div>
      <div style="font-size:0.82rem; color:#546e7a !important; margin-top:4px; text-align:right;">
        Kaggle · CC0 Public Domain
      </div>
    </div>
  </div>
</div>

<!-- ★ SLIDE 3: EDA 핵심 발견 -->
<div class="slide">
  <span class="snum">03 / 06</span>
  <div class="slabel">EDA Findings</div>
  <div class="stitle">데이터가 말해준 것</div>
  <div class="two-col" style="margin-top:16px; align-items:center;">
    <div style="text-align:center;">
      <img src="{imgA}" style="max-width:100%; height:auto;" />
    </div>
    <div style="text-align:center;">
      <img src="{imgB}" style="max-width:100%; height:auto;" />
    </div>
  </div>
  <div class="insight">
    💡 공부 <strong>시간</strong>보다 공부 <strong>꾸준함</strong>이 훨씬 강한 예측 변수
    &nbsp;—&nbsp; '거의 안 함' 그룹의 <strong>74%</strong>가 관리 필요 단계
  </div>
</div>

<!-- ★ SLIDE 4: DEMO -->
<div class="slide">
  <span class="snum">04 / 06</span>
  <div class="slabel">Live Demo</div>
  <div class="demo-box">
    <div class="demo-main">🚀 LIVE DEMO</div>
    <div class="demo-url">http://localhost:8501</div>
  </div>
  <ul class="steps">
    <li><span class="snum-badge">1</span>📋 데이터 개요 — 1,200행·36열 요약 확인</li>
    <li><span class="snum-badge">2</span>🔍 EDA → 타겟 분포 → 공부시간×CGPA 히트맵</li>
    <li><span class="snum-badge">3</span>⚙️ 전처리 → ▶ 실행 → 결과 확인 + CSV 다운로드</li>
    <li><span class="snum-badge">4</span>🧪 심화 분석 → SDL 지수 → 위험도별 박스플롯</li>
    <li><span class="snum-badge">5</span>🎛️ 필터 분석 → 스트레스 슬라이더 조정 → 상관관계 변화 확인</li>
  </ul>
</div>

<!-- ★ SLIDE 5: 특성 & 분석 결과 -->
<div class="slide">
  <span class="snum">05 / 06</span>
  <div class="slabel">Feature Engineering &amp; Results</div>
  <div class="stitle">새로 만든 특성 &amp; 핵심 수치</div>
  <div class="two-col" style="margin-top:16px; align-items:center;">
    <div>
      <div class="slabel" style="margin-bottom:8px;">SDL 자기주도 학습 지수</div>
      <div class="formula">
        SDL = 공부일관성 (0~2)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 복습빈도 (0~3)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 과제제때제출 (0~3)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 과제제출 (0~3)<br>
        <strong style="font-size:1.05rem;">합계 최대 11점</strong>
      </div>
      <table style="width:100%; border-collapse:collapse; font-size:0.9rem; margin-top:12px;">
        <tr style="background:#E3F2FD;">
          <th style="padding:6px 10px; text-align:left; border-radius:4px 0 0 0;">발견</th>
          <th style="padding:6px 10px; text-align:right; border-radius:0 4px 0 0;">수치</th>
        </tr>
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:6px 10px; color:#212121;">공부 일관성 최저 → 관리 필요</td>
          <td style="padding:6px 10px; text-align:right; font-weight:700; color:#C62828;">74%</td>
        </tr>
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:6px 10px; color:#212121;">학업 만족도 최저 → 관리 필요</td>
          <td style="padding:6px 10px; text-align:right; font-weight:700; color:#C62828;">65%</td>
        </tr>
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:6px 10px; color:#212121;">스크린타임 6h+ 생산성 저하</td>
          <td style="padding:6px 10px; text-align:right; font-weight:700; color:#E65100;">−16%</td>
        </tr>
        <tr>
          <td style="padding:6px 10px; color:#212121;">안정 그룹 비율 (클래스 불균형)</td>
          <td style="padding:6px 10px; text-align:right; font-weight:700; color:#1565C0;">15%</td>
        </tr>
      </table>
    </div>
    <div style="text-align:center;">
      <img src="{imgC}" style="max-width:100%; height:auto;" />
    </div>
  </div>
</div>

<!-- ★ SLIDE 6: 결론 & 한계 -->
<div class="slide">
  <span class="snum">06 / 06</span>
  <div class="slabel">Conclusion</div>
  <div class="stitle">결론 &amp; 한계</div>
  <div class="insight" style="margin-top:16px;">
    공부 <strong>꾸준함(일관성)</strong>과 <strong>SDL 자기주도 지수</strong>가
    단순 공부 시간보다 학업 관리 단계를 훨씬 잘 설명한다
  </div>
  <div class="concl-row">
    <div class="ccard cg">
      <h4>✅ 알게 된 것</h4>
      <p>
        꾸준함 &gt; 시간<br>
        스트레스 高 + 에너지 低 = 번아웃<br>
        외부 압박 없음 → 오히려 관리 필요 ↑
      </p>
    </div>
    <div class="ccard co">
      <h4>⚠️ 한계</h4>
      <p>
        안정 15% 클래스 불균형<br>
        자기보고 편향 가능성<br>
        인과관계 아닌 상관관계만 확인
      </p>
    </div>
    <div class="ccard cb">
      <h4>🔭 향후 과제</h4>
      <p>
        Random Forest + SMOTE 모델<br>
        SHAP 변수 중요도 시각화<br>
        LLM 맞춤형 학습 상담 리포트
      </p>
    </div>
  </div>
</div>

</div><!-- /deck -->
""", unsafe_allow_html=True)
