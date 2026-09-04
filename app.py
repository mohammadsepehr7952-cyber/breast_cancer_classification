import base64
import io
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
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
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown ul, .stMarkdown ol,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
            direction: rtl;
            text-align: right;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background("background.png")

st.title("🎗️ پیش‌بینی سرطان سینه با چند مدل کلاسیفیکیشن")
st.caption("پروژه کلاسی‌فیکیشن پیشرفته — SMOTE + Hyperparameter Tuning + Logistic Regression / Naive Bayes / KNN / Decision Tree / Random Forest / XGBoost / LightGBM / CatBoost")

# ---------- توابع کش‌شده (برای جلوگیری از اجرای دوباره‌ی محاسبات سنگین در هر rerun) ----------
@st.cache_data
def load_default_data():
    return pd.read_csv("data.csv")


@st.cache_data
def load_uploaded_data(file_bytes: bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data
def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.copy()
    drop_candidates = [c for c in ["id", "Unnamed: 32"] if c in data.columns]
    data = data.drop(columns=drop_candidates)
    data["target"] = data["diagnosis"].map({"M": 0, "B": 1})
    data = data.drop(columns=["diagnosis"])
    data = data.dropna(subset=["target"])
    data = data.fillna(data.median(numeric_only=True))
    return data


@st.cache_data
def compute_pca(data: pd.DataFrame, feature_cols: list):
    pca = PCA(n_components=2)
    return pca.fit_transform(data[feature_cols])


# ---------- بارگذاری داده ----------
st.sidebar.header("منبع داده")
use_uploaded = st.sidebar.toggle("آپلود فایل CSV دستی (اختیاری)", value=False)

if use_uploaded:
    uploaded_file = st.file_uploader("فایل CSV دیتاست سرطان سینه را آپلود کن", type=["csv"])
    if uploaded_file is None:
        st.info("منتظر آپلود فایل...")
        st.stop()
    try:
        raw_data = load_uploaded_data(uploaded_file.getvalue())
    except Exception:
        st.error("❌ نتونستم فایل رو بخونم. مطمئن شو فایل واقعاً CSV معتبره.")
        st.stop()
else:
    try:
        raw_data = load_default_data()
    except FileNotFoundError:
        st.error("❌ فایل data.csv کنار app.py پیدا نشد. مطمئن شو تو ریپازیتوری گیت‌هابت هست.")
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

# ---------- پاکسازی داده (کش‌شده) ----------
data = clean_data(raw_data)
feature_cols = [c for c in data.columns if c != "target"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 بررسی داده (EDA)", "⚖️ SMOTE و توازن", "🤖 مدل‌سازی",
    "🔮 پیش‌بینی نمونه جدید", "📝 گزارش نهایی"
])

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
    y_raw = data["target"]
    X_pca = compute_pca(data, feature_cols)
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
    st.subheader("آموزش و مقایسه هشت مدل: LR / NB / KNN / Decision Tree / Random Forest / XGBoost / LightGBM / CatBoost")
    st.caption(
        "هر هشت مدل روی همون داده‌ی Train متوازن‌شده با SMOTE آموزش می‌بینن و روی همون Test دست‌نخورده "
        "ارزیابی می‌شن تا مقایسه‌شون منصفانه باشه."
    )

    if st.button("🔍 آموزش و مقایسه هر هشت مدل"):
        with st.spinner("در حال جستجوی بهترین Hyperparameter و آموزش مدل‌ها..."):

            # ---------- Logistic Regression ----------
            param_grid_lr = {"C": [0.01, 0.1, 1, 5, 10, 20, 50]}
            grid_lr = GridSearchCV(
                LogisticRegression(solver="liblinear", random_state=0),
                param_grid_lr, cv=5, scoring="roc_auc"
            )
            grid_lr.fit(X_train_sm, y_train_sm)
            best_C = grid_lr.best_params_["C"]

            lr_model = LogisticRegression(solver="liblinear", C=best_C, random_state=0)
            lr_model.fit(X_train_sm, y_train_sm)
            y_pred_lr = lr_model.predict(X_test_full)
            y_proba_lr = lr_model.predict_proba(X_test_full)[:, 1]
            acc_lr = accuracy_score(y_test_full, y_pred_lr)
            auc_lr = roc_auc_score(y_test_full, y_proba_lr)
            cv_lr = grid_lr.best_score_

            # ---------- Naive Bayes ----------
            nb_model = GaussianNB()
            nb_model.fit(X_train_sm, y_train_sm)
            y_pred_nb = nb_model.predict(X_test_full)
            y_proba_nb = nb_model.predict_proba(X_test_full)[:, 1]
            acc_nb = accuracy_score(y_test_full, y_pred_nb)
            auc_nb = roc_auc_score(y_test_full, y_proba_nb)
            cv_nb = cross_val_score(nb_model, X_train_sm, y_train_sm, cv=KFold(5)).mean()

            # ---------- KNN (نیاز به Scale کردن داده داره) ----------
            scaler = StandardScaler()
            X_train_sm_scaled = scaler.fit_transform(X_train_sm)
            X_test_scaled = scaler.transform(X_test_full)

            param_grid_knn = {"n_neighbors": list(range(1, 21))}
            grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring="roc_auc")
            grid_knn.fit(X_train_sm_scaled, y_train_sm)
            best_k = grid_knn.best_params_["n_neighbors"]

            knn_model = KNeighborsClassifier(n_neighbors=best_k)
            knn_model.fit(X_train_sm_scaled, y_train_sm)
            y_pred_knn = knn_model.predict(X_test_scaled)
            y_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]
            acc_knn = accuracy_score(y_test_full, y_pred_knn)
            auc_knn = roc_auc_score(y_test_full, y_proba_knn)
            cv_knn = grid_knn.best_score_

            # ---------- Decision Tree ----------
            param_grid_dt = {"max_depth": [3, 4, 5, 6, 7, 8, None]}
            grid_dt = GridSearchCV(
                DecisionTreeClassifier(random_state=0),
                param_grid_dt, cv=5, scoring="roc_auc"
            )
            grid_dt.fit(X_train_sm, y_train_sm)
            best_depth_dt = grid_dt.best_params_["max_depth"]

            dt_model = DecisionTreeClassifier(max_depth=best_depth_dt, random_state=0)
            dt_model.fit(X_train_sm, y_train_sm)
            y_pred_dt = dt_model.predict(X_test_full)
            y_proba_dt = dt_model.predict_proba(X_test_full)[:, 1]
            acc_dt = accuracy_score(y_test_full, y_pred_dt)
            auc_dt = roc_auc_score(y_test_full, y_proba_dt)
            cv_dt = grid_dt.best_score_

            # ---------- Random Forest ----------
            param_grid_rf = {"n_estimators": [100, 150, 200], "max_depth": [5, 7, 9]}
            grid_rf = GridSearchCV(
                RandomForestClassifier(random_state=0),
                param_grid_rf, cv=5, scoring="roc_auc"
            )
            grid_rf.fit(X_train_sm, y_train_sm)
            best_params_rf = grid_rf.best_params_

            rf_model = RandomForestClassifier(
                n_estimators=best_params_rf["n_estimators"],
                max_depth=best_params_rf["max_depth"], random_state=0
            )
            rf_model.fit(X_train_sm, y_train_sm)
            y_pred_rf = rf_model.predict(X_test_full)
            y_proba_rf = rf_model.predict_proba(X_test_full)[:, 1]
            acc_rf = accuracy_score(y_test_full, y_pred_rf)
            auc_rf = roc_auc_score(y_test_full, y_proba_rf)
            cv_rf = grid_rf.best_score_

            # ---------- XGBoost ----------
            xgb_model = XGBClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                eval_metric="logloss", random_state=0
            )
            xgb_model.fit(X_train_sm, y_train_sm)
            y_pred_xgb = xgb_model.predict(X_test_full)
            y_proba_xgb = xgb_model.predict_proba(X_test_full)[:, 1]
            acc_xgb = accuracy_score(y_test_full, y_pred_xgb)
            auc_xgb = roc_auc_score(y_test_full, y_proba_xgb)
            cv_xgb = cross_val_score(xgb_model, X_train_sm, y_train_sm, cv=KFold(5)).mean()

            # ---------- LightGBM ----------
            lgbm_model = LGBMClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                random_state=0, verbose=-1
            )
            lgbm_model.fit(X_train_sm, y_train_sm)
            y_pred_lgbm = lgbm_model.predict(X_test_full)
            y_proba_lgbm = lgbm_model.predict_proba(X_test_full)[:, 1]
            acc_lgbm = accuracy_score(y_test_full, y_pred_lgbm)
            auc_lgbm = roc_auc_score(y_test_full, y_proba_lgbm)
            cv_lgbm = cross_val_score(lgbm_model, X_train_sm, y_train_sm, cv=KFold(5)).mean()

            # ---------- CatBoost ----------
            cat_model = CatBoostClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                random_state=0, verbose=0
            )
            cat_model.fit(X_train_sm, y_train_sm)
            y_pred_cat = cat_model.predict(X_test_full)
            y_proba_cat = cat_model.predict_proba(X_test_full)[:, 1]
            acc_cat = accuracy_score(y_test_full, y_pred_cat)
            auc_cat = roc_auc_score(y_test_full, y_proba_cat)
            cv_cat = cross_val_score(cat_model, X_train_sm, y_train_sm, cv=KFold(5)).mean()

            st.session_state["all_models"] = {
                "Logistic Regression": {
                    "model": lr_model, "acc": acc_lr, "auc": auc_lr, "cv": cv_lr,
                    "y_test": y_test_full, "y_pred": y_pred_lr, "y_proba": y_proba_lr,
                    "extra_param": f"C={best_C}", "needs_scaling": False,
                },
                "Naive Bayes": {
                    "model": nb_model, "acc": acc_nb, "auc": auc_nb, "cv": cv_nb,
                    "y_test": y_test_full, "y_pred": y_pred_nb, "y_proba": y_proba_nb,
                    "extra_param": "GaussianNB", "needs_scaling": False,
                },
                "KNN": {
                    "model": knn_model, "acc": acc_knn, "auc": auc_knn, "cv": cv_knn,
                    "y_test": y_test_full, "y_pred": y_pred_knn, "y_proba": y_proba_knn,
                    "extra_param": f"K={best_k}", "needs_scaling": True, "scaler": scaler,
                },
                "Decision Tree": {
                    "model": dt_model, "acc": acc_dt, "auc": auc_dt, "cv": cv_dt,
                    "y_test": y_test_full, "y_pred": y_pred_dt, "y_proba": y_proba_dt,
                    "extra_param": f"max_depth={best_depth_dt}", "needs_scaling": False,
                },
                "Random Forest": {
                    "model": rf_model, "acc": acc_rf, "auc": auc_rf, "cv": cv_rf,
                    "y_test": y_test_full, "y_pred": y_pred_rf, "y_proba": y_proba_rf,
                    "extra_param": f"n_estimators={best_params_rf['n_estimators']}, max_depth={best_params_rf['max_depth']}",
                    "needs_scaling": False,
                },
                "XGBoost": {
                    "model": xgb_model, "acc": acc_xgb, "auc": auc_xgb, "cv": cv_xgb,
                    "y_test": y_test_full, "y_pred": y_pred_xgb, "y_proba": y_proba_xgb,
                    "extra_param": "n_estimators=150, lr=0.1, max_depth=5", "needs_scaling": False,
                },
                "LightGBM": {
                    "model": lgbm_model, "acc": acc_lgbm, "auc": auc_lgbm, "cv": cv_lgbm,
                    "y_test": y_test_full, "y_pred": y_pred_lgbm, "y_proba": y_proba_lgbm,
                    "extra_param": "n_estimators=150, lr=0.1, max_depth=5", "needs_scaling": False,
                },
                "CatBoost": {
                    "model": cat_model, "acc": acc_cat, "auc": auc_cat, "cv": cv_cat,
                    "y_test": y_test_full, "y_pred": y_pred_cat, "y_proba": y_proba_cat,
                    "extra_param": "n_estimators=150, lr=0.1, max_depth=5", "needs_scaling": False,
                },
            }

    if "all_models" in st.session_state:
        models_dict = st.session_state["all_models"]

        st.subheader("جدول مقایسه مدل‌ها")
        comp_df = pd.DataFrame({
            "Model": list(models_dict.keys()),
            "Best Param": [models_dict[m]["extra_param"] for m in models_dict],
            "Accuracy": [round(models_dict[m]["acc"], 4) for m in models_dict],
            "AUC": [round(models_dict[m]["auc"], 4) for m in models_dict],
            "CV Mean": [round(models_dict[m]["cv"], 4) for m in models_dict],
        })
        st.dataframe(comp_df, use_container_width=True)

        best_model_name = comp_df.loc[comp_df["AUC"].idxmax(), "Model"]
        st.success(f"🏆 بهترین مدل بر اساس AUC: **{best_model_name}**")

        st.subheader("جزئیات هر مدل")
        selected_model_name = st.selectbox(
            "کدوم مدل رو با جزئیات ببینی؟", list(models_dict.keys()),
            index=list(models_dict.keys()).index(best_model_name)
        )
        r = models_dict[selected_model_name]

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{r['acc']:.3f}")
        m2.metric("AUC", f"{r['auc']:.3f}")
        m3.metric("میانگین Cross Val", f"{r['cv']:.3f}")

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
        st.info("روی دکمه‌ی بالا بزن تا هر سه مدل آموزش داده بشن و مقایسه بشن.")

# ---------- تب ۴: پیش‌بینی نمونه جدید ----------
with tab4:
    if "all_models" not in st.session_state:
        st.warning("اول باید تو تب «مدل‌سازی» مدل‌ها رو آموزش بدی.")
    else:
        models_dict = st.session_state["all_models"]
        st.subheader("مقادیر ویژگی‌های نمونه‌ی جدید را وارد کن")

        predict_model_name = st.selectbox("با کدوم مدل پیش‌بینی بشه؟", list(models_dict.keys()))
        chosen = models_dict[predict_model_name]
        model = chosen["model"]

        input_vals = {}
        cols = st.columns(3)
        for i, col_name in enumerate(feature_cols):
            default_val = float(data[col_name].median())
            input_vals[col_name] = cols[i % 3].number_input(col_name, value=default_val, format="%.4f")

        if st.button("پیش‌بینی کن"):
            new_sample = np.array([[input_vals[c] for c in feature_cols]])
            if chosen.get("needs_scaling"):
                new_sample = chosen["scaler"].transform(new_sample)

            pred = model.predict(new_sample)[0]
            proba = model.predict_proba(new_sample)[0][1]

            if pred == 0:
                st.error(
                    f"⚠️ احتمال خوش‌خیمی: {proba*100:.1f}% — مدل «{predict_model_name}» "
                    "این نمونه را بدخیم (Malignant) پیش‌بینی می‌کند."
                )
            else:
                st.success(
                    f"✅ احتمال خوش‌خیمی: {proba*100:.1f}% — مدل «{predict_model_name}» "
                    "این نمونه را خوش‌خیم (Benign) پیش‌بینی می‌کند."
                )

# ---------- تب ۵: گزارش نهایی ----------
with tab5:
    st.subheader("📝 گزارش نهایی پروژه")

    if "all_models" not in st.session_state:
        st.warning("برای ساخت گزارش، اول باید تو تب «مدل‌سازی» مدل‌ها رو آموزش بدی.")
    else:
        models_dict = st.session_state["all_models"]
        comp_df = pd.DataFrame({
            "Model": list(models_dict.keys()),
            "Accuracy": [round(models_dict[m]["acc"], 4) for m in models_dict],
            "AUC": [round(models_dict[m]["auc"], 4) for m in models_dict],
            "CV Mean": [round(models_dict[m]["cv"], 4) for m in models_dict],
        })
        best_model_name = comp_df.loc[comp_df["AUC"].idxmax(), "Model"]
        best_row = comp_df[comp_df["Model"] == best_model_name].iloc[0]

        comp_lines = "\n".join(
            f"- **{row['Model']}**: Accuracy = {row['Accuracy']} | AUC = {row['AUC']} | CV Mean = {row['CV Mean']}"
            for _, row in comp_df.iterrows()
        )

        report_text = f"""# گزارش نهایی پروژه پیش‌بینی سرطان سینه

**تاریخ تولید گزارش:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## ۱. خلاصه داده
- تعداد نمونه‌ها: {data.shape[0]}
- تعداد ویژگی‌ها: {len(feature_cols)}
- نسبت کلاس خوش‌خیم (Benign): {(data['target'].mean()*100):.1f}%
- نسبت کلاس بدخیم (Malignant): {(100 - data['target'].mean()*100):.1f}%

## ۲. پیش‌پردازش
- مقادیر گمشده با میانه (median) هر ستون پر شدند.
- داده به نسبت ۷۵٪ آموزش / ۲۵٪ آزمون تقسیم شد (با حفظ نسبت کلاس‌ها - stratify).
- SMOTE فقط روی داده‌ی Train اعمال شد تا کلاس‌ها متوازن بشن، بدون اینکه نشتی داده (Data Leakage) رخ بده.
- برای مدل KNN، داده علاوه بر این با StandardScaler نرمال‌سازی شد چون KNN فاصله‌محوره.

## ۳. مدل‌های آموزش‌دیده و مقایسه
{comp_lines}

## ۴. بهترین مدل
بر اساس معیار AUC، بهترین مدل **{best_model_name}** بود:
- Accuracy: {best_row['Accuracy']}
- AUC: {best_row['AUC']}
- میانگین Cross Validation: {best_row['CV Mean']}

## ۵. جمع‌بندی
در این پروژه هشت الگوریتم کلاسیفیکیشن (Logistic Regression، Naive Bayes، KNN، Decision Tree، Random Forest،
XGBoost، LightGBM، CatBoost) روی دیتاست
Breast Cancer Wisconsin آموزش داده و با معیارهای Accuracy، AUC و Cross Validation مقایسه شدند.
تمام مدل‌ها روی داده‌ی Train متوازن‌شده با SMOTE آموزش دیدند و روی داده‌ی Test دست‌نخورده
ارزیابی شدند تا نتیجه‌ی ارزیابی واقعی و بدون نشتی داده باشد.
"""
        st.markdown(report_text)

        st.download_button(
            label="📥 دانلود گزارش (txt)",
            data=report_text,
            file_name="breast_cancer_report.txt",
            mime="text/plain",
        )
