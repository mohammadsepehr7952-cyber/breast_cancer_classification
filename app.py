import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                              roc_auc_score, accuracy_score)
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

# ---------- بک‌گراند از عکس ----------
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(4,8,20,0.75), rgba(4,8,20,0.75)),
                               url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown {{
            color: #eef2ff !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: rgba(6, 12, 28, 0.88);
        }}
        div[data-testid="stMetric"] {{
            background-color: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 10px;
        }}
        div[data-testid="stTabs"] button {{
            color: #dbe4ff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background("background.png")

st.title("🎗️ پیش‌بینی سرطان سینه با Logistic Regression")
st.caption("پروژه کلاسی‌فیکیشن پیشرفته — SMOTE + Hyperparameter Tuning + AUC")

# ---------- آپلود فایل ----------
uploaded_file = st.file_uploader("فایل CSV دیتاست سرطان سینه را آپلود کن (مثل data.csv)", type=["csv"])

if uploaded_file is None:
    st.info("برای شروع، فایل data.csv را از کامپیوترت آپلود کن.")
    st.stop()

try:
    raw_data = pd.read_csv(uploaded_file)
except Exception:
    st.error("❌ نتونستم فایل رو بخونم. مطمئن شو فایل واقعاً CSV معتبره.")
    st.stop()

# ---------- چک کردن ستون‌های مورد نیاز ----------
required_min_cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]
missing_cols = [c for c in required_min_cols if c not in raw_data.columns]

if missing_cols:
    st.error(
        "❌ فایلی که آپلود کردی فرمت دیتاست سرطان سینه رو نداره.\n\n"
        f"این ستون‌ها توش پیدا نشد: **{', '.join(missing_cols)}**\n\n"
        f"ستون‌های موجود تو فایلت: {', '.join(raw_data.columns[:15])}"
        + (" ..." if len(raw_data.columns) > 15 else "") + "\n\n"
        "لطفاً فایل درست دیتاست Breast Cancer Wisconsin (مثل data.csv) رو آپلود کن."
    )
    st.stop()

if raw_data["diagnosis"].dropna().isin(["M", "B"]).sum() == 0:
    st.error("❌ ستون diagnosis باید مقادیر M (بدخیم) یا B (خوش‌خیم) داشته باشه. فایل آپلودشده این فرمت رو نداره.")
    st.stop()

if raw_data.shape[0] < 20:
    st.warning(f"⚠️ فایل آپلودشده فقط {raw_data.shape[0]} ردیف داره — برای مدل‌سازی داده‌ی کافی نیست.")
    st.stop()

# ---------- پاکسازی داده ----------
data = raw_data.copy()
drop_candidates = [c for c in ["id", "Unnamed: 32"] if c in data.columns]
data = data.drop(columns=drop_candidates)
data["target"] = data["diagnosis"].map({"M": 0, "B": 1})
data = data.drop(columns=["diagnosis"])
data = data.dropna(subset=["target"])
data = data.fillna(data.median(numeric_only=True))

feature_cols = [c for c in data.columns if c != "target"]

tab1, tab2, tab3, tab4 = st.tabs(["📊 بررسی داده (EDA)", "⚖️ SMOTE و توازن", "🤖 مدل‌سازی", "🔮 پیش‌بینی نمونه جدید"])

# ---------- تب ۱: EDA ----------
with tab1:
    st.subheader("نگاه اول به داده")
    st.dataframe(data.head())

    c1, c2, c3 = st.columns(3)
    c1.metric("تعداد نمونه‌ها", data.shape[0])
    c2.metric("تعداد ویژگی‌ها", len(feature_cols))
    c3.metric("نسبت خوش‌خیم (Benign)", f"{(data['target'].mean()*100):.1f}%")

    st.subheader("توزیع کلاس‌ها")
    fig, ax = plt.subplots(figsize=(5, 3))
    sb.countplot(x="target", data=data, ax=ax)
    ax.set_xticklabels(["Malignant (0)", "Benign (1)"])
    st.pyplot(fig)

    st.subheader("۱۰ ویژگی با بیشترین ارتباط به تشخیص")
    top_corr = data.corr(numeric_only=True)["target"].abs().sort_values(ascending=False)[1:11]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    top_corr.plot(kind="barh", color="teal", ax=ax2)
    ax2.invert_yaxis()
    ax2.set_title("Top 10 Features Correlated with Cancer Diagnosis")
    ax2.set_xlabel("Correlation Coefficient (abs)")
    st.pyplot(fig2)

    st.subheader("تصویرسازی PCA (فشرده‌سازی ۳۰ ویژگی به ۲ بعد)")
    X_raw = data[feature_cols]
    y_raw = data["target"]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_raw)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sb.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_raw, palette={0: "red", 1: "green"}, ax=ax3)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("PCA Visualization - Malignant vs Benign")
    ax3.legend(title="Target", labels=["Malignant (0)", "Benign (1)"])
    st.pyplot(fig3)

# ---------- آماده‌سازی X, y ----------
X = data[feature_cols].values
y = data["target"].values

# ---------- تب ۲: SMOTE ----------
with tab2:
    st.subheader("توزیع کلاس‌ها قبل از SMOTE")
    before_counts = pd.Series(y).value_counts().rename({0: "Malignant", 1: "Benign"})
    st.bar_chart(before_counts)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    smote = SMOTE(random_state=0)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_full, y_train_full)

    st.subheader("توزیع کلاس‌ها بعد از SMOTE (فقط روی داده‌ی Train)")
    after_counts = pd.Series(y_train_sm).value_counts().rename({0: "Malignant", 1: "Benign"})
    st.bar_chart(after_counts)

    st.info(
        "⚠️ SMOTE فقط روی داده‌ی Train اعمال می‌شه، نه روی کل داده. "
        "این کار از نشتی داده (Data Leakage) جلوگیری می‌کنه و ارزیابی نهایی رو واقعی نگه می‌داره."
    )

# ---------- تب ۳: مدل‌سازی ----------
with tab3:
    st.subheader("جستجوی خودکار بهترین Hyperparameter (C) با GridSearchCV")

    if st.button("🔍 اجرای GridSearchCV و آموزش مدل"):
        with st.spinner("در حال جستجوی بهترین C و آموزش مدل..."):
            param_grid = {"C": [0.01, 0.1, 1, 5, 10, 20, 50]}
            grid = GridSearchCV(
                LogisticRegression(solver="liblinear", random_state=0),
                param_grid, cv=5, scoring="roc_auc"
            )
            grid.fit(X_train_sm, y_train_sm)
            best_C = grid.best_params_["C"]

            model = LogisticRegression(solver="liblinear", C=best_C, random_state=0)
            model.fit(X_train_sm, y_train_sm)

            y_pred = model.predict(X_test_full)
            y_proba = model.predict_proba(X_test_full)[:, 1]

            acc = accuracy_score(y_test_full, y_pred)
            auc = roc_auc_score(y_test_full, y_proba)

            kf = KFold(10)
            cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=kf)

            st.session_state["model_results"] = {
                "best_C": best_C, "model": model, "acc": acc, "auc": auc,
                "cv_scores": cv_scores, "y_test": y_test_full, "y_pred": y_pred, "y_proba": y_proba,
            }

    if "model_results" in st.session_state:
        r = st.session_state["model_results"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("بهترین C", r["best_C"])
        m2.metric("Accuracy", f"{r['acc']:.3f}")
        m3.metric("AUC", f"{r['auc']:.3f}")
        m4.metric("میانگین Cross Val", f"{r['cv_scores'].mean():.3f}")

        st.subheader("نمودار ROC")
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_proba"])
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        ax4.plot(fpr, tpr, label=f"AUC = {r['auc']:.3f}", color="darkorange")
        ax4.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax4.set_xlabel("False Positive Rate")
        ax4.set_ylabel("True Positive Rate")
        ax4.legend()
        st.pyplot(fig4)

        col_cm, col_cr = st.columns(2)
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(r["y_test"], r["y_pred"])
            fig5, ax5 = plt.subplots(figsize=(4, 4))
            sb.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                       xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"], ax=ax5)
            ax5.set_xlabel("Predicted")
            ax5.set_ylabel("Actual")
            st.pyplot(fig5)

        with col_cr:
            st.subheader("Classification Report")
            report = classification_report(r["y_test"], r["y_pred"],
                                             target_names=["Malignant", "Benign"], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3))
    else:
        st.info("روی دکمه‌ی بالا بزن تا مدل آموزش داده بشه.")

# ---------- تب ۴: پیش‌بینی نمونه جدید ----------
with tab4:
    if "model_results" not in st.session_state:
        st.warning("اول باید تو تب «مدل‌سازی» مدل رو آموزش بدی.")
    else:
        st.subheader("مقادیر ویژگی‌های نمونه‌ی جدید را وارد کن")
        model = st.session_state["model_results"]["model"]

        input_vals = {}
        cols = st.columns(3)
        for i, col_name in enumerate(feature_cols):
            default_val = float(data[col_name].median())
            input_vals[col_name] = cols[i % 3].number_input(col_name, value=default_val, format="%.4f")

        if st.button("پیش‌بینی کن"):
            new_sample = np.array([[input_vals[c] for c in feature_cols]])
            pred = model.predict(new_sample)[0]
            proba = model.predict_proba(new_sample)[0][1]

            if pred == 0:
                st.error(f"⚠️ احتمال خوش‌خیمی: {proba*100:.1f}% — مدل این نمونه را بدخیم (Malignant) پیش‌بینی می‌کند.")
            else:
                st.success(f"✅ احتمال خوش‌خیمی: {proba*100:.1f}% — مدل این نمونه را خوش‌خیم (Benign) پیش‌بینی می‌کند.")
