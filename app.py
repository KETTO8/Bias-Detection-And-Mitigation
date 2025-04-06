# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bias Mitigation in Hiring", layout="wide")
st.title("💼 Bias Mitigation in Hiring - Dashboard")

# Upload data
st.sidebar.header("1. Upload Data")
train_file = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])

if train_file:
    df = pd.read_csv(train_file)
    st.success("✅ Data Loaded Successfully!")
    st.subheader("📈 Sample Data")
    st.dataframe(df.head())

    # Drop income column if present
    if 'income' in df.columns:
        df = df.drop(columns=['income'])

    st.sidebar.header("2. Select Target & Protected Attributes")
    target = st.sidebar.selectbox("Select Target Column", df.columns)
    protected_attrs = st.sidebar.multiselect("Select Protected Attributes", df.columns)

    if target and protected_attrs:
        features = [col for col in df.columns if col not in protected_attrs + [target]]

        # Split
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Summary card
        with st.container():
            st.markdown("### 📊 Summary Card")
            st.markdown(f"- **Model Accuracy**: `{acc:.2f}`")
            st.markdown(f"- **AUC Score**: `{auc:.2f}`")
            st.markdown(f"- **Selected Protected Attributes**: `{', '.join(protected_attrs)}`")
            st.markdown(f"- **Target Variable**: `{target}`")

        st.subheader("🔧 Retrained Logistic Model")
        st.metric("Accuracy", f"{acc:.2f}")
        st.metric("AUC Score", f"{auc:.2f}")

        # Tabs for analysis
        tab1, tab2, tab3 = st.tabs(["Model Metrics", "Fairness Metrics", "Feature Importance"])

        with tab1:
            st.write("**Classification Report**")
            st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

            st.write("**Confusion Matrix**")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

            st.write("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend()
            st.pyplot(fig_roc)

        with tab2:
            st.write("**Fairness Metrics**")
            for attr in protected_attrs:
                metric_frame = MetricFrame(
                    metrics=selection_rate,
                    y_true=y_test,
                    y_pred=y_pred,
                    sensitive_features=df.loc[y_test.index, attr]
                )
                st.write(f"**{attr} Disparate Impact**: {metric_frame.ratio():.2f}")
                fig, ax = plt.subplots()
                metric_frame.by_group.plot(kind='bar', ax=ax, color='coral')
                ax.set_title(f'Selection Rate by {attr}')
                ax.set_ylabel('Selection Rate')
                st.pyplot(fig)

            if st.button("🌀 Mitigate Bias using Exponentiated Gradient"):
                # Combine multiple protected attributes into a single column
                combined_attr = df.loc[X_train.index, protected_attrs].astype(str).agg('-'.join, axis=1)
                combined_test_attr = df.loc[X_test.index, protected_attrs].astype(str).agg('-'.join, axis=1)

                mitigator = ExponentiatedGradient(
                    estimator=LogisticRegression(solver="liblinear"),
                    constraints=DemographicParity()
                )
                mitigator.fit(X_train, y_train, sensitive_features=combined_attr)
                y_mitigated_pred = mitigator.predict(X_test)

                st.write("### 🔄 Post-Mitigation Metrics")
                mitigated_acc = accuracy_score(y_test, y_mitigated_pred)
                st.metric("Mitigated Accuracy", f"{mitigated_acc:.2f}")

                metric_frame_mitigated = MetricFrame(
                    metrics=selection_rate,
                    y_true=y_test,
                    y_pred=y_mitigated_pred,
                    sensitive_features=combined_test_attr
                )
                mitigated_di = metric_frame_mitigated.ratio()
                st.write(f"**Post-Mitigation Disparate Impact**: {mitigated_di:.2f}")

                fig, ax = plt.subplots()
                metric_frame_mitigated.by_group.plot(kind='bar', ax=ax, color='green')
                ax.set_title(f'Post-Mitigation Selection Rate by Combined Attributes')
                ax.set_ylabel('Selection Rate')
                st.pyplot(fig)

        with tab3:
            st.write("**Feature Importance (Coefficients)**")
            coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
            fig, ax = plt.subplots()
            coefs.plot(kind='barh', ax=ax, color='teal')
            ax.set_title('Logistic Regression Coefficients')
            st.pyplot(fig)

else:
    st.info("👈 Upload a CSV file to begin.")
