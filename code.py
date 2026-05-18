import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

st.set_page_config(page_title="Dự đoán GPA AI", layout="wide")

# ================= LOAD DATA & TRAIN MODEL =================
@st.cache_resource
def load_model():
    # Đọc file
    df = pd.read_csv("GPA - Trang tính1.csv")
    
    # Mapping cho các cột phân loại
    mapping = {
        # Attendance
        "Trên 90%": 2,
        # Learning style
        "Tự học một mình": 0,
        "Kết hợp cả hai": 1,
        "Học nhóm": 2,
        # Part-time job
        "Không đi làm": 0,
        "Làm dưới 16h/tuần": 1,
        "Làm trên 16h/tuần": 2,
        # Club
        "Không": 0,
        "Có": 1,
        # Sleep
        "Dưới 5 tiếng": 0,
        "5 - 7 tiếng": 1,
        "Trên 7 tiếng": 2,
        # Social media
        "Dưới 2 tiếng": 0,
        "2 - 4 tiếng": 1,
        "4 - 6 tiếng": 2,
        "Trên 6 tiếng": 3,
    }
    
    df = df.replace(mapping)
    
    # Target mapping
    gpa_map = {
        "Dưới 2.0": 0,
        "2.0 - 2.5": 1,
        "2.6 - 3.0": 2,
        "3.1 - 3.5": 3,
        "3.5 - 4.0": 4,   # Thêm vì dữ liệu có
    }
    
    df["GPA_label"] = df["GPA học kỳ gần nhất của bạn là bao nhiêu? (Thang 4)"].map(gpa_map)
    
    # Features
    X = df.drop([
        "GPA học kỳ gần nhất của bạn là bao nhiêu? (Thang 4)", 
        "GPA_label"
    ], axis=1)
    
    y = df["GPA_label"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, X.columns.tolist(), acc, gpa_map, df

model, feature_names, accuracy, gpa_map, original_df = load_model()

# ================= UI =================
st.title("🎓 Dự Đoán GPA Bằng AI")
st.markdown("**Dựa trên thói quen học tập của sinh viên**")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("🕒 Giờ tự học mỗi tuần", 0, 40, 10)
    num_subjects = st.slider("📚 Số môn đăng ký", 1, 12, 5)
    attendance = st.selectbox("📍 Tỷ lệ đi học", ["Trên 90%"])
    learning_style = st.selectbox("📖 Hình thức học", 
                                 ["Tự học một mình", "Kết hợp cả hai", "Học nhóm"])

with col2:
    job = st.selectbox("💼 Làm thêm", 
                      ["Không đi làm", "Làm dưới 16h/tuần", "Làm trên 16h/tuần"])
    club = st.selectbox("🎯 Tham gia CLB/Đoàn", ["Không", "Có"])
    sleep = st.selectbox("😴 Giờ ngủ mỗi đêm", 
                        ["Dưới 5 tiếng", "5 - 7 tiếng", "Trên 7 tiếng"])
    social = st.selectbox("📱 Thời gian dùng MXH mỗi ngày", 
                         ["Dưới 2 tiếng", "2 - 4 tiếng", "4 - 6 tiếng", "Trên 6 tiếng"])

# Tạo input DataFrame
input_dict = {
    feature_names[0]: study_hours,
    feature_names[1]: num_subjects,
    feature_names[2]: mapping.get(attendance, 2),
    feature_names[3]: mapping.get(learning_style, 1),
    feature_names[4]: mapping.get(job, 0),
    feature_names[5]: mapping.get(club, 0),
    feature_names[6]: mapping.get(sleep, 1),
    feature_names[7]: mapping.get(social, 1),
}

input_df = pd.DataFrame([input_dict])[feature_names]

# ================= PREDICTION =================
if st.button("🚀 Dự đoán GPA của tôi", type="primary", use_container_width=True):
    with st.spinner("AI đang phân tích..."):
        time.sleep(1.1)
        pred_label = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        
    inv_gpa = {v: k for k, v in gpa_map.items()}
    predicted_gpa = inv_gpa[pred_label]
    confidence = proba.max()
    
    st.success(f"**GPA dự đoán: {predicted_gpa}**")
    st.progress(float(confidence))
    st.caption(f"Độ tin cậy: **{confidence:.1%}**")

# ================= METRICS =================
st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Độ chính xác mô hình", f"{accuracy:.1%}")
c2.metric("Số mẫu dữ liệu", len(original_df))
c3.metric("Số features", len(feature_names))

st.caption("💡 *Model được huấn luyện trên dữ liệu thực tế của sinh viên. Kết quả chỉ mang tính tham khảo.*")
