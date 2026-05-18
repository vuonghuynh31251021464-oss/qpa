import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

st.set_page_config(page_title="GPA AI - UEH", layout="wide")

# ================= MAPPING =================
mapping = {
    "Trên 90%": 2, "Tự học một mình": 0, "Kết hợp cả hai": 1, "Học nhóm": 2,
    "Không đi làm": 0, "Làm dưới 16h/tuần": 1, "Làm trên 16h/tuần": 2,
    "Không": 0, "Có": 1,
    "Dưới 5 tiếng": 0, "5 - 7 tiếng": 1, "Trên 7 tiếng": 2,
    "Dưới 2 tiếng": 0, "2 - 4 tiếng": 1, "4 - 6 tiếng": 2, "Trên 6 tiếng": 3,
}

gpa_map = {"Dưới 2.0": 0, "2.0 - 2.5": 1, "2.6 - 3.0": 2, "3.1 - 3.5": 3, "3.5 - 4.0": 4}

@st.cache_resource
def train_model():
    df = pd.read_csv("GPA - Trang tính1.csv")
    df = df.replace(mapping)
    df["GPA_label"] = df.iloc[:, -1].map(gpa_map)
    
    X = df.iloc[:, :-2]
    y = df["GPA_label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns.tolist(), acc, df, X_train, X_test, y_train, y_test

model, feature_names, acc, df_raw, X_train, X_test, y_train, y_test = train_model()

# ================= SIDEBAR =================
st.sidebar.title("🎓 GPA AI - UEH")
page = st.sidebar.radio("Chọn chức năng", 
                       ["🤖 Dự đoán GPA", "📊 Phân tích dữ liệu khảo sát"])

# ================= TRANG DỰ ĐOÁN =================
if page == "🤖 Dự đoán GPA":
    st.title("🎓 Dự Đoán GPA Bằng AI")
    st.markdown("**Khảo sát hơn 200 sinh viên UEH**")
    
    col1, col2 = st.columns(2)
    with col1:
        study = st.slider("🕒 Giờ tự học mỗi tuần", 0, 40, 10)
        subjects = st.slider("📚 Số môn học", 1, 12, 6)
        attendance = st.selectbox("📍 Tỷ lệ đi học", ["Trên 90%"])
        learning = st.selectbox("📖 Hình thức học", ["Tự học một mình", "Kết hợp cả hai", "Học nhóm"])

    with col2:
        job = st.selectbox("💼 Làm thêm", ["Không đi làm", "Làm dưới 16h/tuần", "Làm trên 16h/tuần"])
        club = st.selectbox("🎯 Tham gia CLB", ["Không", "Có"])
        sleep = st.selectbox("😴 Giờ ngủ mỗi đêm", ["Dưới 5 tiếng", "5 - 7 tiếng", "Trên 7 tiếng"])
        social = st.selectbox("📱 Thời gian MXH", ["Dưới 2 tiếng", "2 - 4 tiếng", "4 - 6 tiếng", "Trên 6 tiếng"])

    input_data = pd.DataFrame([{
        feature_names[0]: study, feature_names[1]: subjects,
        feature_names[2]: mapping[attendance], feature_names[3]: mapping[learning],
        feature_names[4]: mapping[job], feature_names[5]: mapping[club],
        feature_names[6]: mapping[sleep], feature_names[7]: mapping[social],
    }])

    if st.button("🚀 Dự đoán GPA", type="primary", use_container_width=True):
        with st.spinner("AI đang phân tích..."):
            time.sleep(1.2)
            pred = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0].max()
        
        result_map = {v: k for k, v in gpa_map.items()}
        st.success(f"**🎓 GPA dự đoán: {result_map[pred]}**")
        st.progress(float(proba))
        st.caption(f"Độ tin cậy: **{proba:.1%}**")

    st.divider()
    st.metric("Độ chính xác mô hình", f"{acc:.2%}")

# ================= TRANG PHÂN TÍCH DỮ LIỆU =================
else:
    st.title("📊 Phân tích dữ liệu khảo sát")
    st.subheader("Hơn 200 sinh viên UEH")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Tổng quan", "📊 Biểu đồ GPA", "🔥 Heatmap", "📋 Train/Test Split"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng số sinh viên", len(df_raw))
        col2.metric("Số features", len(feature_names))
        col3.metric("Độ chính xác mô hình", f"{acc:.2%}")
        
        st.write("**Phân bố GPA:**")
        st.bar_chart(df_raw['GPA học kỳ gần nhất của bạn là bao nhiêu? (Thang 4)'].value_counts())

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df_raw, y='GPA học kỳ gần nhất của bạn là bao nhiêu? (Thang 4)', 
                     order=df_raw['GPA học kỳ gần nhất của bạn là bao nhiêu? (Thang 4)'].value_counts().index, ax=ax)
        ax.set_title("Phân bố GPA của sinh viên UEH")
        st.pyplot(fig)

        st.subheader("Giờ tự học theo GPA")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df_raw, x='GPA học kỳ gần nhất của bạn là bao nhiêu? (Thang 4)', 
                   y=df_raw.columns[0], ax=ax2)
        st.pyplot(fig2)

    with tab3:
        st.subheader("Heatmap Correlation")
        # Tạo bản sao số hóa
        df_num = df_raw.replace(mapping).select_dtypes(include=['int64', 'float64'])
        corr = df_num.corr()
        
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax3)
        st.pyplot(fig3)

    with tab4:
        st.subheader("Train / Test Split")
        col_a, col_b = st.columns(2)
        col_a.metric("Số mẫu Train", len(X_train))
        col_b.metric("Số mẫu Test", len(X_test))
        
        st.write("**Phân bố GPA trong Train set:**")
        st.bar_chart(pd.Series(y_train).value_counts().sort_index())
        
        st.write("**Phân bố GPA trong Test set:**")
        st.bar_chart(pd.Series(y_test).value_counts().sort_index())

    st.caption("Dữ liệu từ khảo sát thực tế hơn 200 sinh viên UEH")
