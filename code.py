# ================= IMPORT =================
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ================= LOAD DATA =================
df_gpa = pd.read_csv("GPA - Trang tính1.csv")

# ================= MAPPING =================
mapping = {
    "Dưới 5 tiếng": 0,
    "5 - 7 tiếng": 1,
    "Trên 7 tiếng": 2,

    "Dưới 2 tiếng": 0,
    "2 - 4 tiếng": 1,
    "4 - 6 tiếng": 2,

    "Không đi làm": 0,
    "Làm dưới 16h/tuần": 1,
    "Làm trên 16h/tuần": 2,

    "Không": 0,
    "Có": 1,

    "Tự học một mình": 0,
    "Kết hợp cả hai": 1,

    "Trên 90%": 2
}

df_gpa = df_gpa.replace(mapping)

# ================= ENCODE LABEL =================
gpa_map = {
    "Dưới 2.0": 0,
    "2.0 - 2.5": 1,
    "2.6 - 3.0": 2,
    "3.1 - 3.5": 3,
    "Trên 3.5": 4
}

df_gpa["GPA_label"] = df_gpa[
    "GPA học kỳ gần nhất của bạn là bao nhiêu? (Thang 4)"
].map(gpa_map)

# ================= FEATURE =================
X_gpa = df_gpa.drop(
    ["GPA học kỳ gần nhất của bạn là bao nhiêu? (Thang 4)", "GPA_label"],
    axis=1
)

y_gpa = df_gpa["GPA_label"]

# ================= TRAIN =================
X_train_gpa, X_test_gpa, y_train_gpa, y_test_gpa = train_test_split(
    X_gpa, y_gpa, test_size=0.2, random_state=42
)

model_gpa = RandomForestClassifier(n_estimators=120, random_state=42)
model_gpa.fit(X_train_gpa, y_train_gpa)

# ================= ĐÁNH GIÁ =================
y_pred_gpa = model_gpa.predict(X_test_gpa)
acc = accuracy_score(y_test_gpa, y_pred_gpa)

# ================= MENU =================
st.sidebar.markdown("---")
if st.sidebar.button("🎓 Dự đoán GPA"):
    st.session_state.page = "gpa"

# ================= UI =================
if "page" in st.session_state and st.session_state.page == "gpa":

    st.header("🎓 Dự đoán GPA bằng AI")

    col1, col2 = st.columns(2)

    with col1:
        study = st.slider("Giờ tự học / tuần", 0, 40, 10)
        subjects = st.slider("Số môn học", 1, 10, 5)
        attendance = st.selectbox("Đi học", ["Trên 90%"])
        learning = st.selectbox("Cách học", ["Tự học một mình", "Kết hợp cả hai"])

    with col2:
        job = st.selectbox("Làm thêm", ["Không đi làm", "Làm dưới 16h/tuần", "Làm trên 16h/tuần"])
        club = st.selectbox("CLB", ["Không", "Có"])
        sleep = st.selectbox("Ngủ", ["Dưới 5 tiếng", "5 - 7 tiếng", "Trên 7 tiếng"])
        social = st.selectbox("MXH", ["Dưới 2 tiếng", "2 - 4 tiếng", "4 - 6 tiếng"])

    # ================= INPUT =================
    input_dict_gpa = {
        X_gpa.columns[0]: study,
        X_gpa.columns[1]: subjects,
        X_gpa.columns[2]: mapping[attendance],
        X_gpa.columns[3]: mapping[learning],
        X_gpa.columns[4]: mapping[job],
        X_gpa.columns[5]: mapping[club],
        X_gpa.columns[6]: mapping[sleep],
        X_gpa.columns[7]: mapping[social],
    }

    input_gpa = pd.DataFrame([input_dict_gpa])

    # ✅ FIX LỖI FEATURE
    input_gpa = input_gpa.reindex(columns=X_gpa.columns, fill_value=0)

    # ================= PREDICT =================
    if st.button("🚀 Dự đoán GPA"):

        with st.spinner("🤖 AI đang phân tích học lực..."):
            import time
            time.sleep(1.2)

            pred = model_gpa.predict(input_gpa)[0]

        inv_map = {v: k for k, v in gpa_map.items()}
        result = inv_map[pred]

        st.success(f"🎓 GPA dự đoán: {result}")

    # ================= METRIC =================
    st.subheader("📊 Độ chính xác model")
    st.metric("Accuracy", f"{acc:.2f}")
