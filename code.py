import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# ================= PAGE CONFIG =================
st.set_page_config(page_title="GPA AI - UEH", layout="wide", page_icon="🎓")

# ================= UEH GENZ THEME CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg:           #0f1e14;
    --bg-card:      #162b1d;
    --bg-card2:     #1c3526;
    --border:       #2a5038;
    --green-bright: #3dba68;
    --green-mid:    #2d8c4e;
    --green-dark:   #1a5c2a;
    --green-text:   #7de0a4;
    --orange:       #f5841f;
    --orange-soft:  #f5a94e;
    --orange-dim:   #3d2408;
    --text-main:    #e8f5ee;
    --text-sub:     #8fbfa0;
    --white:        #ffffff;
}

html, body, [class*="css"], * {
    font-family: 'Be Vietnam Pro', sans-serif !important;
}
.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
}
.block-container {
    background-color: var(--bg) !important;
    padding-top: 1.5rem !important;
}
p, span, div, li, td, th, label,
[data-testid="stMarkdownContainer"] * {
    color: var(--text-main) !important;
}
h1, h2, h3, h4, h5, h6 {
    color: var(--green-text) !important;
    font-weight: 800 !important;
}

.hero-banner {
    background: linear-gradient(135deg, #0d3d1e 0%, var(--green-mid) 60%, var(--orange) 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border);
}
.hero-banner::before {
    content: "UEH";
    position: absolute;
    right: -10px;
    top: -20px;
    font-size: 9rem;
    font-weight: 900;
    color: rgba(255,255,255,0.05);
    letter-spacing: -4px;
    pointer-events: none;
}
.hero-banner h1 {
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    color: #ffffff !important;
    margin: 0 0 0.3rem 0 !important;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.hero-banner p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 1rem !important;
    margin: 0 !important;
    font-weight: 500 !important;
}
.hero-badge {
    display: inline-block;
    background: var(--orange);
    color: #fff !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 0.7rem;
}

.ueh-card {
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.ueh-card h4 {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: var(--green-text) !important;
    margin: 0 0 0.8rem 0 !important;
}

[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
}
[data-testid="metric-container"] label {
    color: var(--text-sub) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--green-text) !important;
    font-size: 1.9rem !important;
    font-weight: 900 !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--green-mid), var(--orange)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    box-shadow: 0 4px 20px rgba(245,132,31,0.4) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(245,132,31,0.55) !important;
}
.stButton > button {
    background: var(--bg-card2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
}

[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--text-main) !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: var(--green-bright) !important;
    box-shadow: 0 0 0 3px rgba(61,186,104,0.2) !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: var(--text-main) !important;
}
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {
    background: var(--bg-card2) !important;
    color: var(--text-main) !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover {
    background: var(--border) !important;
}

[data-testid="stSlider"] label {
    color: var(--text-main) !important;
    font-weight: 600 !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--green-bright), var(--orange)) !important;
    border-radius: 999px !important;
}
.stProgress > div > div {
    background: var(--bg-card2) !important;
    border-radius: 999px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    border: 1.5px solid var(--border) !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-weight: 700 !important;
    color: var(--text-sub) !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.2s !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--green-mid), var(--green-dark)) !important;
    color: #fff !important;
}

[data-testid="stSidebar"] {
    background: #0a1710 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-main) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    color: var(--green-text) !important;
    font-size: 1.3rem !important;
    font-weight: 900 !important;
    border-bottom: 2px solid var(--orange) !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1rem !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    padding: 0.5rem 0.8rem !important;
    transition: background 0.2s !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) {
    background: var(--orange) !important;
}

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

.stCaption, [data-testid="stCaptionContainer"] * {
    color: var(--text-sub) !important;
    font-size: 0.8rem !important;
}
.stSpinner > div { border-top-color: var(--orange) !important; }

label {
    font-weight: 700 !important;
    color: var(--text-main) !important;
    font-size: 0.88rem !important;
}

[data-testid="stAlert"] {
    background: rgba(61,186,104,0.12) !important;
    border-left: 4px solid var(--green-bright) !important;
    border-radius: 12px !important;
}
[data-testid="stAlert"] * {
    color: var(--green-text) !important;
    font-weight: 600 !important;
}

.gpa-result-chip {
    display: inline-block;
    background: linear-gradient(135deg, var(--green-mid), var(--green-dark));
    color: #fff !important;
    font-size: 1.7rem;
    font-weight: 900;
    padding: 0.7rem 2.2rem;
    border-radius: 50px;
    margin: 0.5rem 0;
    box-shadow: 0 6px 24px rgba(45,140,78,0.45);
    border: 1px solid var(--green-bright);
}
.confidence-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--orange-soft) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}
.acc-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--orange-dim);
    border: 1.5px solid var(--orange);
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-weight: 700;
    font-size: 0.9rem;
    color: var(--orange-soft) !important;
}
.attendance-note {
    background: rgba(245,132,31,0.1);
    border: 1px solid rgba(245,132,31,0.4);
    border-radius: 10px;
    padding: 0.5rem 0.9rem;
    font-size: 0.82rem;
    color: var(--orange-soft) !important;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ================= MAPPING =================
# Attendance: ordinal encoding — càng đi học nhiều càng cao
# Dưới 50% = 0, 50-75% = 1, 75-90% = 2, Trên 90% = 3
attendance_map = {
    "Dưới 50%":  0,
    "50% - 75%": 1,
    "75% - 90%": 2,
    "Trên 90%":  3,
}

mapping = {
    # attendance
    "Dưới 50%":  0,
    "50% - 75%": 1,
    "75% - 90%": 2,
    "Trên 90%":  3,
    # learning style
    "Tự học một mình": 0,
    "Kết hợp cả hai":  1,
    "Học nhóm":        2,
    # job
    "Không đi làm":       0,
    "Làm dưới 16h/tuần":  1,
    "Làm trên 16h/tuần":  2,
    # club
    "Không": 0,
    "Có":    1,
    # sleep
    "Dưới 5 tiếng": 0,
    "5 - 7 tiếng":  1,
    "Trên 7 tiếng": 2,
    # social
    "Dưới 2 tiếng": 0,
    "2 - 4 tiếng":  1,
    "4 - 6 tiếng":  2,
    "Trên 6 tiếng": 3,
}

gpa_map = {
    "Dưới 2.0":  0,
    "2.0 - 2.5": 1,
    "2.6 - 3.0": 2,
    "3.1 - 3.5": 3,
    "3.5 - 4.0": 4,
}

UEH_GREEN   = "#2d8c4e"
UEH_ORANGE  = "#f5841f"
UEH_DARK    = "#1a5c2a"
UEH_PALE    = "#d4f5e0"
UEH_PALETTE = [UEH_GREEN, UEH_ORANGE, "#3dba68", "#f5a94e", "#1a5c2a", "#c0600a"]


def ueh_style_plot():
    plt.rcParams.update({
        "figure.facecolor":  "#f7faf8",
        "axes.facecolor":    "#ffffff",
        "axes.edgecolor":    "#d0e8d8",
        "axes.labelcolor":   "#0e1f16",
        "axes.titleweight":  "bold",
        "axes.titlecolor":   "#1a5c2a",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.color":       "#5a7065",
        "ytick.color":       "#5a7065",
        "grid.color":        "#e5f0ea",
        "grid.linestyle":    "--",
        "grid.alpha":        0.7,
        "font.family":       "DejaVu Sans",
        "text.color":        "#0e1f16",
    })


def augment_low_attendance(df_real: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Tạo dữ liệu synthetic cho các mức attendance thấp hơn (không có trong CSV gốc).
    Logic:
      - 75-90%  → GPA bị kéo xuống ~1 bậc so với bản gốc (xác suất 55%)
      - 50-75%  → GPA bị kéo xuống ~1-2 bậc (xác suất 70%)
      - Dưới 50%→ GPA bị kéo xuống ~2-3 bậc (xác suất 85%), phần lớn rơi vào Dưới 2.0 / 2.0-2.5
    """
    gpa_col = df_real.columns[-1]   # cột GPA gốc (chưa encode)
    att_col = df_real.columns[2]    # cột attendance

    gpa_ordered = ["Dưới 2.0", "2.0 - 2.5", "2.6 - 3.0", "3.1 - 3.5", "3.5 - 4.0"]

    configs = [
        # (label_mới, n_samples, prob_giảm, mức_giảm_tối_đa)
        ("75% - 90%", 60,  0.55, 1),
        ("50% - 75%", 60,  0.70, 2),
        ("Dưới 50%",  60,  0.85, 3),
    ]

    rows = []
    base = df_real.copy()

    for att_label, n, prob_drop, max_drop in configs:
        sample = base.sample(n=n, replace=True, random_state=rng.integers(0, 9999))
        for _, row in sample.iterrows():
            new_row = row.copy()
            new_row[att_col] = att_label
            orig_gpa = row[gpa_col]
            if orig_gpa in gpa_ordered:
                idx = gpa_ordered.index(orig_gpa)
                if rng.random() < prob_drop:
                    drop = int(rng.integers(1, max_drop + 1))
                    new_idx = max(0, idx - drop)
                    new_row[gpa_col] = gpa_ordered[new_idx]
            rows.append(new_row)

    return pd.concat([df_real, pd.DataFrame(rows)], ignore_index=True)


@st.cache_resource
def train_model():
    df_real = pd.read_csv("GPA - Trang tính1.csv")

    # Augment dữ liệu
    rng = np.random.default_rng(42)
    df_aug = augment_low_attendance(df_real, rng)

    # Encode
    df_enc = df_aug.copy()
    att_col  = df_enc.columns[2]
    gpa_col  = df_enc.columns[-1]

    df_enc[att_col] = df_enc[att_col].map(attendance_map)

    for col in df_enc.columns[:-1]:
        if df_enc[col].dtype == object:
            df_enc[col] = df_enc[col].map(mapping)

    df_enc["GPA_label"] = df_enc[gpa_col].map(gpa_map)
    df_enc = df_enc.dropna(subset=["GPA_label"])
    df_enc["GPA_label"] = df_enc["GPA_label"].astype(int)

    X = df_enc.iloc[:, :-2]
    y = df_enc["GPA_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", max_depth=12
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    return model, X.columns.tolist(), acc, df_aug, X_train, X_test, y_train, y_test


model, feature_names, acc, df_raw, X_train, X_test, y_train, y_test = train_model()

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("# 🎓 GPA AI · UEH")
    st.markdown("<hr style='border-color:#3dba68;margin:0.5rem 0 1rem 0'/>", unsafe_allow_html=True)
    page = st.radio(
        "Chọn chức năng",
        ["🤖 Dự đoán GPA", "📊 Phân tích dữ liệu khảo sát"],
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""<div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:0.8rem 1rem;
            font-size:0.82rem;color:#c8e6d0'>
        📌 Dữ liệu từ <b>200+ sinh viên</b> UEH<br>
        🔄 Augmented với logic attendance<br>
        🎯 Model accuracy: <b style='color:#f5841f'>{acc:.1%}</b>
        </div>""",
        unsafe_allow_html=True,
    )

# ================= TRANG DỰ ĐOÁN =================
if page == "🤖 Dự đoán GPA":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">✦ AI-Powered</div>
        <h1>Dự Đoán GPA Bằng AI 🚀</h1>
        <p>Nhập thông tin học tập của bạn — AI sẽ dự đoán GPA dựa trên khảo sát 200+ sinh viên UEH</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="ueh-card"><h4>📚 Học tập & Thời gian</h4>', unsafe_allow_html=True)
        study = st.slider("🕒 Giờ tự học mỗi tuần", 0, 40, 10)
        subjects = st.slider("📚 Số môn học", 1, 12, 6)

        attendance = st.selectbox(
            "📍 Tỷ lệ đi học đầy đủ trên lớp",
            ["Trên 90%", "75% - 90%", "50% - 75%", "Dưới 50%"],
        )

        # Ghi chú động theo mức attendance
        att_notes = {
            "Trên 90%":  ("✅", "Tuyệt vời! Đi học đầy đủ là nền tảng vững chắc cho GPA cao.", "#3dba68"),
            "75% - 90%": ("🟡", "Khá tốt. Bỏ một số buổi có thể ảnh hưởng nhỏ đến kết quả.", "#f5c842"),
            "50% - 75%": ("🟠", "Cần cải thiện. Tỷ lệ vắng cao làm giảm đáng kể cơ hội GPA tốt.", "#f5841f"),
            "Dưới 50%":  ("🔴", "Rủi ro cao! Vắng quá nhiều thường dẫn đến GPA thấp hoặc trượt môn.", "#e84040"),
        }
        icon, note, color = att_notes[attendance]
        st.markdown(
            f"<div class='attendance-note' style='border-color:{color}40;color:{color} !important'>"
            f"{icon} {note}</div>",
            unsafe_allow_html=True,
        )

        learning = st.selectbox("📖 Hình thức học", ["Tự học một mình", "Kết hợp cả hai", "Học nhóm"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ueh-card"><h4>💼 Lối sống & Sinh hoạt</h4>', unsafe_allow_html=True)
        job   = st.selectbox("💼 Làm thêm", ["Không đi làm", "Làm dưới 16h/tuần", "Làm trên 16h/tuần"])
        club  = st.selectbox("🎯 Tham gia CLB", ["Không", "Có"])
        sleep = st.selectbox("😴 Giờ ngủ mỗi đêm", ["Dưới 5 tiếng", "5 - 7 tiếng", "Trên 7 tiếng"])
        social = st.selectbox("📱 Thời gian MXH / ngày",
                              ["Dưới 2 tiếng", "2 - 4 tiếng", "4 - 6 tiếng", "Trên 6 tiếng"])
        st.markdown('</div>', unsafe_allow_html=True)

    input_data = pd.DataFrame([{
        feature_names[0]: study,
        feature_names[1]: subjects,
        feature_names[2]: attendance_map[attendance],
        feature_names[3]: mapping[learning],
        feature_names[4]: mapping[job],
        feature_names[5]: mapping[club],
        feature_names[6]: mapping[sleep],
        feature_names[7]: mapping[social],
    }])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Dự đoán GPA ngay!", type="primary", use_container_width=True):
        with st.spinner("AI đang phân tích hành vi học tập của bạn..."):
            time.sleep(1.2)
            pred  = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0].max()

        result_map = {v: k for k, v in gpa_map.items()}
        gpa_label  = result_map[pred]

        # Màu chip theo GPA
        chip_colors = {
            "Dưới 2.0":  ("linear-gradient(135deg,#7a1a1a,#c0392b)", "#e84040"),
            "2.0 - 2.5": ("linear-gradient(135deg,#7a4210,#c0600a)", "#f5841f"),
            "2.6 - 3.0": ("linear-gradient(135deg,#7a6d10,#c0a80a)", "#f5c842"),
            "3.1 - 3.5": ("linear-gradient(135deg,#1a5c2a,#2d8c4e)", "#3dba68"),
            "3.5 - 4.0": ("linear-gradient(135deg,#0d3d1e,#2d8c4e)", "#7de0a4"),
        }
        grad, border = chip_colors.get(gpa_label, (
            "linear-gradient(135deg,#1a5c2a,#2d8c4e)", "#3dba68"))

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#f7faf8,#d4f5e0);border:2px solid #3dba68;
                    border-radius:18px;padding:1.5rem 2rem;text-align:center;margin-top:1rem'>
            <div style='font-size:0.78rem;font-weight:700;letter-spacing:1.5px;
                        text-transform:uppercase;color:#2d8c4e;margin-bottom:0.4rem'>
                🎯 Kết quả dự đoán
            </div>
            <div style='display:inline-block;background:{grad};color:#fff;
                        font-size:1.7rem;font-weight:900;padding:0.7rem 2.2rem;
                        border-radius:50px;margin:0.5rem 0;
                        box-shadow:0 6px 24px rgba(45,140,78,0.45);
                        border:1px solid {border}'>
                {gpa_label}
            </div>
            <div class='confidence-label'>Độ tin cậy: {proba:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Phân phối xác suất tất cả các lớp
        st.markdown("<br>", unsafe_allow_html=True)
        probas = model.predict_proba(input_data)[0]
        classes = [result_map[i] for i in range(len(probas))]

        st.markdown("**📊 Phân phối xác suất các mức GPA:**")
        for cls, p in zip(classes, probas):
            col_label, col_bar = st.columns([2, 5])
            col_label.markdown(
                f"<span style='font-size:0.82rem;font-weight:600'>{cls}</span>",
                unsafe_allow_html=True,
            )
            col_bar.progress(float(p), text=f"{p:.1%}")

    st.divider()
    st.markdown(
        f"<div class='acc-badge'>📊 Độ chính xác mô hình &nbsp;·&nbsp; "
        f"<span style='font-size:1.05rem'>{acc:.2%}</span></div>",
        unsafe_allow_html=True,
    )

# ================= TRANG PHÂN TÍCH DỮ LIỆU =================
else:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">✦ Data Insights</div>
        <h1>Phân Tích Dữ Liệu Khảo Sát 📊</h1>
        <p>Khám phá patterns học tập của hơn 200 sinh viên UEH (bao gồm dữ liệu augmented)</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Tổng quan", "📊 Biểu đồ GPA", "🎓 Attendance vs GPA", "🔥 Heatmap", "📋 Train/Test Split"]
    )

    ueh_style_plot()
    gpa_col = df_raw.columns[-1]
    att_col = df_raw.columns[2]

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("👥 Tổng sinh viên (augmented)", len(df_raw))
        c2.metric("🔢 Số features", len(feature_names))
        c3.metric("🎯 Model accuracy", f"{acc:.2%}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Phân bố GPA toàn bộ khảo sát")

        gpa_counts = df_raw[gpa_col].value_counts()
        fig0, ax0 = plt.subplots(figsize=(9, 3.5))
        bars = ax0.bar(gpa_counts.index, gpa_counts.values,
                       color=UEH_PALETTE[:len(gpa_counts)],
                       edgecolor="white", linewidth=1.5, zorder=3)
        ax0.grid(axis="y", zorder=0)
        ax0.set_title("Phân bố GPA - Sinh viên UEH (Augmented)", pad=12)
        ax0.set_ylabel("Số sinh viên")
        for bar in bars:
            ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(int(bar.get_height())), ha='center', va='bottom',
                     fontweight='bold', fontsize=10, color=UEH_DARK)
        ax0.spines[['top', 'right']].set_visible(False)
        fig0.tight_layout()
        st.pyplot(fig0)

    with tab2:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        gpa_order = df_raw[gpa_col].value_counts().index
        palette1  = {cat: UEH_PALETTE[i % len(UEH_PALETTE)] for i, cat in enumerate(gpa_order)}
        sns.countplot(data=df_raw, y=gpa_col, order=gpa_order,
                      palette=palette1, ax=ax1, edgecolor="white", linewidth=1)
        ax1.set_title("Phân bố GPA của sinh viên UEH", pad=12)
        ax1.set_xlabel("Số sinh viên")
        ax1.set_ylabel("")
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.grid(axis="x", zorder=0)
        fig1.tight_layout()
        st.pyplot(fig1)

        st.markdown("#### ⏱️ Giờ tự học theo nhóm GPA")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bp = ax2.boxplot(
            [df_raw[df_raw[gpa_col] == cat][df_raw.columns[0]].dropna()
             for cat in sorted(df_raw[gpa_col].unique())],
            labels=sorted(df_raw[gpa_col].unique()),
            patch_artist=True, notch=False,
            medianprops=dict(color=UEH_ORANGE, linewidth=2.5),
            whiskerprops=dict(color=UEH_DARK),
            capprops=dict(color=UEH_DARK),
            flierprops=dict(marker='o', color=UEH_ORANGE, alpha=0.5, markersize=5),
        )
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(UEH_PALE if i % 2 == 0 else "#fde8cc")
            patch.set_edgecolor(UEH_GREEN)
            patch.set_linewidth(1.5)
        ax2.set_title("Giờ tự học theo nhóm GPA", pad=12)
        ax2.set_xlabel("Nhóm GPA")
        ax2.set_ylabel("Số giờ / tuần")
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.grid(axis="y", zorder=0)
        fig2.tight_layout()
        st.pyplot(fig2)

    with tab3:
        st.markdown("#### 🎓 Tỷ lệ đi học và GPA")

        att_order = ["Dưới 50%", "50% - 75%", "75% - 90%", "Trên 90%"]
        gpa_order2 = ["Dưới 2.0", "2.0 - 2.5", "2.6 - 3.0", "3.1 - 3.5", "3.5 - 4.0"]

        # Stacked bar: attendance vs GPA distribution
        cross = pd.crosstab(df_raw[att_col], df_raw[gpa_col])
        cross = cross.reindex(index=[a for a in att_order if a in cross.index],
                              columns=[g for g in gpa_order2 if g in cross.columns],
                              fill_value=0)
        cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

        fig5, ax5 = plt.subplots(figsize=(11, 5))
        bottom = np.zeros(len(cross_pct))
        colors_gpa = ["#e84040", "#f5841f", "#f5c842", "#3dba68", "#1a5c2a"]
        for i, col in enumerate(cross_pct.columns):
            ax5.bar(cross_pct.index, cross_pct[col], bottom=bottom,
                    label=col, color=colors_gpa[i], edgecolor="white", linewidth=0.8)
            bottom += cross_pct[col].values

        ax5.set_title("Phân bố GPA theo Tỷ lệ Đi học (%)", pad=12)
        ax5.set_xlabel("Tỷ lệ đi học")
        ax5.set_ylabel("Tỷ lệ sinh viên (%)")
        ax5.legend(title="GPA", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        ax5.spines[['top', 'right']].set_visible(False)
        fig5.tight_layout()
        st.pyplot(fig5)

        # Count per attendance level
        st.markdown("#### 📊 Số lượng sinh viên theo tỷ lệ đi học")
        att_counts = df_raw[att_col].value_counts().reindex(att_order, fill_value=0)
        fig6, ax6 = plt.subplots(figsize=(8, 3.5))
        att_colors = ["#e84040", "#f5841f", "#f5c842", "#3dba68"]
        bars6 = ax6.bar(att_counts.index, att_counts.values,
                        color=att_colors, edgecolor="white", linewidth=1.5, zorder=3)
        ax6.grid(axis="y", zorder=0)
        ax6.set_title("Số sinh viên theo tỷ lệ đi học (bao gồm augmented)", pad=10)
        ax6.set_ylabel("Số sinh viên")
        for bar in bars6:
            ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(int(bar.get_height())), ha='center', va='bottom',
                     fontweight='bold', fontsize=10, color=UEH_DARK)
        ax6.spines[['top', 'right']].set_visible(False)
        fig6.tight_layout()
        st.pyplot(fig6)

        st.caption("ℹ️ Dữ liệu khảo sát gốc chỉ có nhóm 'Trên 90%'. Các nhóm còn lại được tạo bằng logic augmentation có kiểm soát để huấn luyện mô hình nhận biết tác động của tỷ lệ điểm danh đến GPA.")

    with tab4:
        st.markdown("#### 🔥 Heatmap tương quan các biến")
        df_enc2 = df_raw.copy()
        df_enc2[att_col] = df_enc2[att_col].map(attendance_map)
        for col in df_enc2.columns:
            if df_enc2[col].dtype == object:
                df_enc2[col] = df_enc2[col].map(mapping)
        df_num = df_enc2.select_dtypes(include=['int64', 'float64', 'int32'])
        corr = df_num.corr()

        fig3, ax3 = plt.subplots(figsize=(10, 8))
        cmap = sns.diverging_palette(140, 25, s=85, l=50, as_cmap=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                    linewidths=0.5, linecolor="#e5f0ea",
                    annot_kws={"size": 8.5, "weight": "bold"},
                    ax=ax3, cbar_kws={"shrink": 0.8})
        ax3.set_title("Correlation Heatmap — UEH Student Survey", pad=14)
        ax3.tick_params(axis='x', rotation=35, labelsize=8)
        ax3.tick_params(axis='y', rotation=0, labelsize=8)
        fig3.tight_layout()
        st.pyplot(fig3)

    with tab5:
        st.markdown("#### 📋 Phân chia Train / Test")
        ca, cb = st.columns(2)
        ca.metric("🏋️ Train samples", len(X_train))
        cb.metric("🧪 Test samples", len(X_test))

        st.markdown("<br>", unsafe_allow_html=True)
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4.5))

        for ax, series, title in [
            (ax4a, pd.Series(y_train).value_counts().sort_index(), "Train set"),
            (ax4b, pd.Series(y_test).value_counts().sort_index(), "Test set"),
        ]:
            colors = [UEH_GREEN if i % 2 == 0 else UEH_ORANGE for i in range(len(series))]
            bars = ax.bar(series.index, series.values, color=colors,
                          edgecolor="white", linewidth=1.5, zorder=3)
            ax.set_title(f"Phân bố GPA — {title}", pad=10)
            ax.set_xlabel("GPA label")
            ax.set_ylabel("Số mẫu")
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis="y", zorder=0)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(int(bar.get_height())), ha='center', va='bottom',
                        fontweight='bold', fontsize=9, color=UEH_DARK)
            green_patch  = mpatches.Patch(color=UEH_GREEN, label='Even labels')
            orange_patch = mpatches.Patch(color=UEH_ORANGE, label='Odd labels')
            ax.legend(handles=[green_patch, orange_patch], fontsize=8)

        fig4.tight_layout()
        st.pyplot(fig4)

        st.caption("Dữ liệu từ khảo sát thực tế 200+ sinh viên UEH + augmented data · Powered by Random Forest AI")
