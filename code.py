import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

st.set_page_config(page_title="Dự đoán GPA", layout="centered")

@st.cache_resource
def train_model():
    df = pd.read_csv("GPA - Trang tính1.csv")
    
    # Mapping
    mapping = {
        "Trên 90%": 2,
        "Tự học một mình": 0,
        "Kết hợp cả hai": 1,
        "Học nhóm": 2,
        "Không đi làm": 0,
        "Làm dưới 16h/tuần": 1,
        "Làm trên 16h/tuần": 2,
        "Không": 0,
        "Có": 1,
        "Dưới 5 tiếng": 0,
        "5 - 7 tiếng": 1,
        "Trên 7 tiếng": 2,
        "Dưới 2 tiếng": 0,
        "2 - 4 tiếng": 1,
        "4 - 6 tiếng": 2,
        "Trên 6 tiếng": 3,
    }
    
    df = df.replace(mapping)
    
    # GPA Label
    gpa_map = {
        "Dưới 2.0": 0,
        "2.0 - 2.5": 1,
        "2.6 - 3.0": 2,
        "3.1 - 3.5": 3,
        "3.5 - 4.0": 4
    }
    
    df["GPA_label"] = df.iloc[:, -1].map(gpa_map)
    
    X = df.iloc[:, :-2]      # Tất cả cột trừ 2 cột GPA
    y = df["GPA_label"]
    
    # Train test split - ĐÃ SỬA LỖI stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25, 
        random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, X.columns.tolist(), acc, gpa_map


# ================= TRAIN MODEL =================
model, feature_names, acc, gpa_map = train_model()

# ================= UI =================
st.title("🎓 Dự Đoán GPA Bằng AI")
st.markdown("**Dựa trên thói quen học tập của sinh viên**")

col1, col2 = st.columns(2)

with col1:
    study = st.slider("🕒 Giờ tự học mỗi tuần", 0, 40, 10)
    subjects = st.slider("📚 Số môn học", 1, 12, 6)
    attendance = st.selectbox("📍 Tỷ lệ đi học", ["Trên 90%"])
    learning = st.selectbox("📖 Hình thức học", 
                           ["Tự học một mình", "Kết hợp cả hai", "Học nhóm"])

with col2:
    job = st.selectbox("💼 Làm thêm", 
                      ["Không đi làm", "Làm dưới 16h/tuần", "Làm trên 16h/tuần"])
    club = st.selectbox("🎯 Tham gia CLB", ["Không", "Có"])
    sleep = st.selectbox("😴 Giờ ngủ mỗi đêm", 
                        ["Dưới 5 tiếng", "5 - 7 tiếng", "Trên 7 tiếng"])
    social = st.selectbox("📱 Thời gian MXH mỗi ngày", 
                         ["Dưới 2 tiếng", "2 - 4 tiếng", "4 - 6 tiếng", "Trên 6 tiếng"])

# Tạo input
input_data = pd.DataFrame([{
    feature_names[0]: study,
    feature_names[1]: subjects,
    feature_names[2]: mapping[attendance],
    feature_names[3]: mapping[learning],
    feature_names[4]: mapping[job],
    feature_names[5]: mapping[club],
    feature_names[6]: mapping[sleep],
    feature_names[7]: mapping[social],
}])

# ================= PREDICT =================
if st.button("🚀 Dự đoán GPA", type="primary", use_container_width=True):
    with st.spinner("🤖 AI đang phân tích..."):
        time.sleep(1.2)
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0].max()
    
    result_map = {v: k for k, v in gpa_map.items()}
    st.success(f"**🎓 GPA dự đoán: {result_map[pred]}**")
    st.progress(float(proba))
    st.caption(f"Độ tin cậy: **{proba:.1%}**")

# ================= METRICS =================
st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Độ chính xác mô hình", f"{acc:.2%}")
col2.metric("Số mẫu dữ liệu", "≈ 200")
col3.metric("Thuật toán", "Random Forest")

st.caption("💡 Kết quả chỉ mang tính tham khảo. Hãy cố gắng học tập đều đặn!")
