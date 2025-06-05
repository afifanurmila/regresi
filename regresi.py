import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import kstest
from scipy.stats import norm

st.title("Regresi Linier Berganda dengan Uji Asumsi")

# Upload file
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.dataframe(df)

    # Pilih kolom numerik
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_columns) < 2:
        st.warning("Data harus memiliki setidaknya dua kolom numerik.")
    else:
        y_col = st.selectbox("Pilih variabel Y (dependen):", numeric_columns)
        x_cols = st.multiselect("Pilih variabel X (independen):", [col for col in numeric_columns if col != y_col])

        if y_col and x_cols:
            data = df[[y_col] + x_cols].dropna()
            X = data[x_cols]
            y = data[y_col]

            # Tambahkan konstanta untuk OLS
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()

            st.subheader("Summary Regresi:")
            summary_html = model.summary().as_html()
            st.markdown(f"<div style='overflow-x:auto'>{summary_html}</div>", unsafe_allow_html=True)

            residuals = model.resid

            # -------------------------
            # Uji Normalitas (Kolmogorov-Smirnov)
            # -------------------------
            st.subheader("Uji Asumsi: Normalitas Residual (Kolmogorov-Smirnov)")
            standardized_residuals = (residuals - residuals.mean()) / residuals.std()
            ks_stat, ks_p = kstest(standardized_residuals, 'norm')
            st.write(f"Kolmogorov-Smirnov p-value: {ks_p:.4f}")
            if ks_p > 0.05:
                st.success("Residual berdistribusi normal (p > 0.05)")
            else:
                st.warning("Residual tidak berdistribusi normal (p ≤ 0.05)")

            fig1, ax1 = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax1)
            ax1.set_title("Histogram Residual")
            st.pyplot(fig1)

            # -------------------------
            # Multikolinearitas (VIF)
            # -------------------------
            st.subheader("Uji Asumsi: Multikolinearitas (VIF)")
            X_vif = sm.add_constant(X)
            vif_data = pd.DataFrame()
            vif_data["Variabel"] = X_vif.columns
            vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
            st.dataframe(vif_data[vif_data["Variabel"] != "const"])

            # -------------------------
            # Autokorelasi (Durbin-Watson)
            # -------------------------
            st.subheader("Uji Asumsi: Autokorelasi (Durbin-Watson)")
            dw_stat = durbin_watson(residuals)
            st.write(f"Durbin-Watson statistic: {dw_stat:.4f}")
            if 1.5 < dw_stat < 2.5:
                st.success("Tidak ada autokorelasi yang signifikan")
            else:
                st.warning("Kemungkinan ada autokorelasi")

            # -------------------------
            # Heteroskedastisitas (Uji Glejser)
            # -------------------------
            st.subheader("Uji Asumsi: Homoskedastisitas (Uji Glejser)")
            abs_resid = np.abs(residuals)
            glejser_model = sm.OLS(abs_resid, X_const).fit()
            glejser_html = glejser_model.summary().as_html()
            st.markdown(f"<div style='overflow-x:auto'>{glejser_html}</div>", unsafe_allow_html=True)

            glejser_pvalues = glejser_model.pvalues.drop("const", errors="ignore")
            if all(glejser_pvalues > 0.05):
                st.success("Tidak ada heteroskedastisitas (semua p > 0.05)")
            else:
                st.warning("Ada indikasi heteroskedastisitas (setidaknya satu p ≤ 0.05)")

            # -------------------------
            # Visualisasi Residual
            # -------------------------
            st.subheader("Plot Residual vs Fitted")
            fig2, ax2 = plt.subplots()
            sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, ax=ax2, line_kws={"color": "red"})
            ax2.set_xlabel("Fitted Values")
            ax2.set_ylabel("Residuals")
            st.pyplot(fig2)