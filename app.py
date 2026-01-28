import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="final_prep EDA", layout="wide")

# -----------------------------
# Load data (엑셀 헤더 꼬임 대응)
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=1)

    # 첫 행이 영문 변수명인 패턴 처리
    if str(raw.iloc[0, 0]).strip() == "sido":
        raw.columns = raw.iloc[0].astype(str).str.strip()
        df = raw.iloc[1:].copy()
    else:
        df = raw.copy()

    df.columns = pd.Index(df.columns).astype(str).str.strip()

    # 타입 정리
    for c in ["sido", "sigungu", "region_dummies"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # 숫자형 변환(가능한 것 전부)
    exclude = {"sido", "sigungu", "region_dummies", "sigungu_code", "_Unique"}
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data("final_prep.xlsx")

st.title("final_prep EDA Dashboard")

# -----------------------------
# Sidebar controls
# -----------------------------
exclude = {"sido", "sigungu", "region_dummies", "sigungu_code", "_Unique", "year"}
num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

st.sidebar.header("Controls")
var = st.sidebar.selectbox("Variable", num_cols, index=0 if num_cols else None)

agg = st.sidebar.selectbox("Aggregation", ["mean", "sum"], index=0)
year_min = int(np.nanmin(df["year"])) if "year" in df.columns else None
year_max = int(np.nanmax(df["year"])) if "year" in df.columns else None
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max)) if year_min and year_max else None

top_sido_n = st.sidebar.slider("Top N 시도(라인)", 3, 17, 7)

# 필터
if year_range:
    df_f = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])].copy()
else:
    df_f = df.copy()

# Plotly 한글 폰트 (Windows/Mac/Linux에서 가능한 폴백)
FONT_FAMILY = "Malgun Gothic, Apple SD Gothic Neo, NanumGothic, Noto Sans KR, sans-serif"

def apply_font(fig):
    fig.update_layout(font=dict(family=FONT_FAMILY))
    return fig

# -----------------------------
# Layout
# -----------------------------
if not var:
    st.warning("수치형 변수가 없습니다.")
    st.stop()

col1, col2 = st.columns([1.2, 1])

# -----------------------------
# (1) National time series
# -----------------------------
with col1:
    st.subheader(f"전국 시계열: {var} ({agg})")

    if agg == "mean":
        ts = df_f.groupby("year")[var].mean()
    else:
        ts = df_f.groupby("year")[var].sum(min_count=1)

    ts = ts.dropna()
    fig = px.line(ts, x=ts.index, y=ts.values, markers=True, labels={"x": "year", "y": var})
    fig = apply_font(fig)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# (2) Distribution
# -----------------------------
with col2:
    st.subheader(f"분포: {var}")
    d = df_f[var].dropna()
    fig = px.histogram(d, nbins=30, labels={"value": var}, title=None)
    fig = apply_font(fig)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# (3) Top N sido time series
# -----------------------------
st.subheader(f"시도별 시계열(Top {top_sido_n}): {var} ({agg})")
rank = df_f.groupby("sido")[var].mean().sort_values(ascending=False)
top_sidos = rank.head(top_sido_n).index

g = df_f[df_f["sido"].isin(top_sidos)].copy()
if agg == "mean":
    ts_sido = g.groupby(["year", "sido"])[var].mean().reset_index()
else:
    ts_sido = g.groupby(["year", "sido"])[var].sum(min_count=1).reset_index()

fig = px.line(ts_sido, x="year", y=var, color="sido", markers=False)
fig = apply_font(fig)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# (4) Sido x Year grid (heatmap)
# -----------------------------
st.subheader(f"시도×연도 그리드(Heatmap): {var} ({agg})")

if agg == "mean":
    pv = df_f.pivot_table(index="sido", columns="year", values=var, aggfunc="mean")
else:
    pv = df_f.pivot_table(index="sido", columns="year", values=var, aggfunc="sum")

pv = pv.sort_index().sort_index(axis=1)

# 값 heatmap
fig = px.imshow(
    pv,
    aspect="auto",
    labels=dict(x="year", y="sido", color=var),
)
fig = apply_font(fig)
st.plotly_chart(fig, use_container_width=True)

# 관측치 heatmap (결측이 많은 변수 점검용)
st.subheader(f"시도×연도 관측치 개수(Heatmap): {var}")
cnt = df_f.pivot_table(index="sido", columns="year", values=var, aggfunc="count")
cnt = cnt.reindex(pv.index)

fig = px.imshow(
    cnt,
    aspect="auto",
    labels=dict(x="year", y="sido", color="count"),
)
fig = apply_font(fig)
st.plotly_chart(fig, use_container_width=True)
