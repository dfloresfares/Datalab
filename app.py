import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# 0. INTENTAR IMPORTAR XGBOOST
# ==========================================================
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception as e:
    XGB_AVAILABLE = False


# ==========================================================
# 1. FEATURE ENGINEERING
# ==========================================================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Data_No" in df.columns and "Time" in df.columns:
        df = df.sort_values(["Data_No", "Time"])
    elif "Time" in df.columns:
        df = df.sort_values(["Time"])

    df["dp_diff"] = df["Differential_pressure"].diff().fillna(0.0)
    df["dp_slope"] = df["dp_diff"] / (df["Time"].diff().replace(0, np.nan))
    df["dp_slope"] = df["dp_slope"].replace([np.inf, -np.inf], 0).fillna(0.0)

    df["dp_rolling_mean"] = df["Differential_pressure"].rolling(5, min_periods=1).mean()
    df["dp_rolling_std"] = df["Differential_pressure"].rolling(5, min_periods=2).std().fillna(0.0)

    eps = 1e-3
    df["dp_over_flow"] = df["Differential_pressure"] / (df["Flow_rate"] + eps)

    return df


# ==========================================================
# 2. PREPARAR FEATURES
# ==========================================================
def prepare_features(df, target_col="RUL"):
    df = df.copy()

    if "Dust" in df.columns:
        df = pd.get_dummies(df, columns=["Dust"], drop_first=True)

    candidate_cols = [
        "Differential_pressure", "Flow_rate", "Time", "Dust_feed",
        "dp_diff", "dp_slope", "dp_rolling_mean", "dp_rolling_std", "dp_over_flow"
    ]

    final_cols = [c for c in candidate_cols if c in df.columns]
    final_cols += [c for c in df.columns if c.startswith("Dust_")]

    X = df[final_cols].values
    y = df[target_col].values

    return X, y, final_cols


# ==========================================================
# 3. ENTRENAR Y COMPARAR MODELOS
# ==========================================================
def train_and_evaluate_models(X, y, include_xgb=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    if include_xgb and XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42
        )

    results = []
    predictions = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
        r2 = r2_score(y_test, y_pred)

        results.append({"Modelo": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        predictions[name] = {"y_test": y_test, "y_pred": y_pred}
        fitted_models[name] = model

    metrics_df = pd.DataFrame(results).sort_values(by="RMSE")
    return metrics_df, predictions, fitted_models, scaler



# ==========================================================
# 4. STREAMLIT APP
# ==========================================================
st.set_page_config(page_title="Comparación de modelos para Vida Útil Remanente", page_icon="⚙️")

st.title("⚙️ Estimación de Vida Útil Remanente (RUL) con ML")
st.write("Subí tu base de datos estructurada y compará la performance de distintos modelos supervisados para predecir su vida útil remanente.")

uploaded_file = st.file_uploader("📂 Subí un archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    if "RUL" not in df.columns:
        st.error("El dataset necesita la columna 'RUL'.")
        st.stop()

    st.subheader("🔧 Generando features derivados…")
    df_fe = add_derived_features(df)

    X, y, cols = prepare_features(df_fe)
    st.info(f"Total de features utilizados: {len(cols)}")

    st.subheader("🚀 Entrenando y evaluando modelos…")
    metrics_df, preds, models, scaler = train_and_evaluate_models(X, y, include_xgb=True)

    st.subheader("📊 Comparación de modelos")
    st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}))

    best = metrics_df.iloc[0]["Modelo"]
    st.success(f"🏆 Mejor modelo según RMSE: **{best}**")

    st.subheader("🔍 Vida real vs vida predicha (scatter)")

    y_test = preds[best]["y_test"]
    y_pred = preds[best]["y_pred"]

    corr = np.corrcoef(y_test, y_pred)[0, 1]
    st.write(f"Correlación: **{corr:.3f}**")

    scatter_df = pd.DataFrame({
        "Real": y_test,
        "Predicha": y_pred
        })

    points = (
        alt.Chart(scatter_df)
        .mark_circle(size=40, opacity=0.25, color="#1f77b4")
        .encode(
            x=alt.X("Real:Q", title="Vida real del filtro (RUL)"),
            y=alt.Y("Predicha:Q", title="Vida predicha por el modelo"),
            tooltip=["Real", "Predicha"]
       )
    )

    diagonal = (
        alt.Chart(scatter_df)
        .mark_line(color="red", opacity=0.8)
        .encode(
            x="Real:Q",
            y="Real:Q"
        )
    )

    chart = (points + diagonal).properties(
        width="container",
        height=400,
        title="Relación entre vida real y vida predicha"
    )

    st.altair_chart(chart, use_container_width=True)
    st.subheader("🚦 Semáforo de riesgo por filtro")

    # Usamos el mejor modelo entrenado
    best_model = models[best]

    # Volvemos a escalar TODO el dataset y predecimos RUL para todas las filas
    X_scaled_full = scaler.transform(X)
    rul_pred_full = best_model.predict(X_scaled_full)

    # Copiamos el dataframe con features para poder agregar predicciones
    df_risk = df_fe.copy()
    df_risk["RUL_pred"] = rul_pred_full

    # Tomamos solo el "estado actual" de cada filtro:
    # última medición temporal por Data_No
    if "Data_No" in df_risk.columns and "Time" in df_risk.columns:
        df_risk_sorted = df_risk.sort_values(["Data_No", "Time"])
        idx_last = df_risk_sorted.groupby("Data_No")["Time"].idxmax()
        df_current = df_risk_sorted.loc[idx_last].copy()
    else:
        # fallback: última fila por Data_No si no hay Time limpio
        if "Data_No" in df_risk.columns:
            df_current = df_risk.sort_values("Data_No").groupby("Data_No").tail(1).copy()
        else:
            st.warning("No se encontró la columna 'Data_No'. No se puede armar el semáforo por filtro.")
            df_current = None

    if df_current is not None:

        max_rul_real = df["RUL"].max()
        df_current["RUL_pct"] = df_current["RUL_pred"] / max_rul_real * 100

        def clasificar_estado(pct):
            if pct < 10:
                return "Rojo"
            elif pct < 30:
                return "Amarillo"
            else:
                return "Verde"

        df_current["Estado"] = df_current["RUL_pct"].apply(clasificar_estado)

        n_filtros = len(df_current)
        n_rojo = (df_current["Estado"] == "Rojo").sum()
        n_amarillo = (df_current["Estado"] == "Amarillo").sum()
        n_verde = (df_current["Estado"] == "Verde").sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total filtros", n_filtros)
        col2.metric("🟢 Verdes", n_verde)
        col3.metric("🟡 Amarillos", n_amarillo)
        col4.metric("🔴 Rojos", n_rojo)

        st.markdown("#### 🔴 Filtros en estado ROJO (programar limpieza / intervención)")

        df_rojo = df_current[df_current["Estado"] == "Rojo"].copy()

        if df_rojo.empty:
            st.success("No hay filtros en estado rojo según el umbral configurado (< 10% de vida útil remanente).")
        else:
            cols_to_show = ["Data_No", "RUL_pred", "RUL_pct", "Estado"]
            extras = [c for c in ["Differential_pressure", "Flow_rate", "Time"] if c in df_rojo.columns]
            cols_to_show += extras

            df_rojo = df_rojo[cols_to_show].sort_values("RUL_pred")
            st.dataframe(df_rojo.style.format({"RUL_pred": "{:.1f}", "RUL_pct": "{:.1f}"}))

        with st.expander("Ver tabla completa de estado por filtro"):
            cols_full = ["Data_No", "RUL_pred", "RUL_pct", "Estado"]
            extras_full = [c for c in ["Differential_pressure", "Flow_rate", "Time"] if c in df_current.columns]
            cols_full += extras_full
            st.dataframe(df_current[cols_full].sort_values("RUL_pct"))



    
else:
    st.info("Subí un archivo CSV para comenzar.")

