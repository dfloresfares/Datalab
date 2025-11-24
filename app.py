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
# 0. XGBoost opcional
# ==========================================================
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ==========================================================
# UX: CONFIG IN SIDEBAR
# ==========================================================
st.sidebar.header("⚙️ Configuración")

modo_experto = st.sidebar.checkbox("Modo experto (ML Avanzado)", value=False)
incluir_xgb = st.sidebar.checkbox("Incluir XGBoost", value=False)
umbral_rojo = st.sidebar.slider("Umbral ROJO (%)", min_value=1, max_value=20, value=5)
umbral_amarillo = st.sidebar.slider("Umbral AMARILLO (%)", min_value=umbral_rojo, max_value=50, value=30)

st.sidebar.markdown("---")
usar_demo = st.sidebar.checkbox("🚀 Usar modo demo (dataset de ejemplo)", value=True, key="usar_demo")

if st.sidebar.button("🔄 Resetear demostración"):
    # Forzamos modo demo y recargamos la app
    st.session_state["usar_demo"] = True
    st.experimental_rerun()

# Ruta del dataset demo (en la raíz del repo)
DEMO_PATH = "Entrenamiento_Data_CSV.csv"

# ==========================================================
# 1. Feature engineering
# ==========================================================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Orden
    if "Data_No" in df.columns and "Time" in df.columns:
        df = df.sort_values(["Data_No", "Time"])
    elif "Time" in df.columns:
        df = df.sort_values("Time")

    # ΔP diff + slope
    df["dp_diff"] = df["Differential_pressure"].diff().fillna(0.0)

    if "Time" in df.columns:
        time_diff = df["Time"].diff().replace(0, np.nan)
        df["dp_slope"] = (df["dp_diff"] / time_diff).replace([np.inf, -np.inf], 0).fillna(0.0)
    else:
        df["dp_slope"] = 0.0

    # Rolling
    df["dp_rolling_mean"] = df["Differential_pressure"].rolling(5, min_periods=1).mean()
    df["dp_rolling_std"] = df["Differential_pressure"].rolling(5, min_periods=2).std().fillna(0.0)

    # ΔP/Flow
    eps = 1e-3
    df["dp_over_flow"] = df["Differential_pressure"] / (df["Flow_rate"] + eps)

    return df

# ==========================================================
# 2. Preparar features
# ==========================================================
def prepare_features(df: pd.DataFrame, target_col="RUL"):
    df = df.copy()

    if "Dust" in df.columns:
        df = pd.get_dummies(df, columns=["Dust"], drop_first=True)

    base_cols = [
        "Differential_pressure", "Flow_rate", "Time", "Dust_feed",
        "dp_diff", "dp_slope", "dp_rolling_mean", "dp_rolling_std", "dp_over_flow"
    ]

    feature_cols = [c for c in base_cols if c in df.columns]
    feature_cols += [c for c in df.columns if c.startswith("Dust_")]

    X = df[feature_cols].values
    y = df[target_col].values

    return X, y, feature_cols

# ==========================================================
# 3. Entrenar y evaluar modelos
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
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=42
        ),
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

    results, preds, fitted_models = [], {}, {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
        r2 = r2_score(y_test, y_pred)

        results.append({"Modelo": name, "MAE": mae, "RMSE": rmse, "R2": r2})

        preds[name] = {"y_test": y_test, "y_pred": y_pred}
        fitted_models[name] = model

    metrics_df = pd.DataFrame(results).sort_values(by="RMSE")
    return metrics_df, preds, fitted_models, scaler

# ==========================================================
# 4. LAYOUT PRINCIPAL UX 2.0
# ==========================================================
st.title("⚙️ Estimación de Vida Útil Remanente (RUL) en Filtros Industriales")
st.write("""
Esta aplicación te permite:
1) Cargar tus datos o usar un dataset de ejemplo  
2) Comparar modelos de Machine Learning - En modo normal te devolverá el mejor modelo, en modo avanzado agregará un cuadro comparativo entre modelos. 
3) Ver qué filtros están en **riesgo operativo** (semáforo) - Estos serán los filtros que habrá que intervenir - Se puede crear un flujo automatizado para generación de órdenes de trabajo.
""")

# 4.0 Carga de datos (modo demo o modo normal)
if st.session_state.get("usar_demo", True):
    st.success("Modo demo activado — usando dataset de ejemplo incluido en la app.")
    try:
        df = pd.read_csv(DEMO_PATH)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo demo en la ruta: {DEMO_PATH}")
        st.stop()
else:
    uploaded_file = st.file_uploader(
        "📂 Subí un archivo CSV (por ejemplo, Entrenamiento_Data_CSV.csv)",
        type=["csv"]
    )
    if uploaded_file is None:
        st.info("Subí un archivo para comenzar o activá el modo demo en la barra lateral.")
        st.stop()
    df = pd.read_csv(uploaded_file)

# 4.1 Cargar y mostrar dataset
st.success(f"Archivo cargado correctamente: {df.shape[0]} filas")

st.subheader("👀 Vista rápida del dataset - A continuación, se muestran las primeras 5 filas de tu base de datos:")
st.dataframe(df.head())

if "RUL" not in df.columns:
    st.error("El dataset debe contener la columna 'RUL'")
    st.stop()

# 4.2 Features
df_fe = add_derived_features(df)
X, y, feature_cols = prepare_features(df_fe)

# 4.3 Entrenar
metrics_df, preds, models, scaler = train_and_evaluate_models(
    X, y, include_xgb=incluir_xgb
)

# 4.4 Modelo ganador — KPI
best = metrics_df.iloc[0]["Modelo"]

st.markdown("## 🏆 Modelo ganador (resumen rápido)")
col1, col2, col3 = st.columns(3)
col1.metric("Modelo", best)
col2.metric("RMSE", f"{metrics_df.iloc[0]['RMSE']:.2f}")
corr_best = np.corrcoef(
    preds[best]["y_test"], preds[best]["y_pred"]
)[0, 1]
col3.metric("Correlación", f"{corr_best:.3f}")

# 4.5 Scatter
st.subheader("🔍 Relación entre Vida Real y Predicha")

scatter_df = pd.DataFrame({
    "Real": preds[best]["y_test"],
    "Predicha": preds[best]["y_pred"]
})

points = (
    alt.Chart(scatter_df)
    .mark_circle(size=40, opacity=0.25, color="#1f77b4")
    .encode(
        x=alt.X("Real:Q", title="Vida real"),
        y=alt.Y("Predicha:Q", title="Vida predicha"),
        tooltip=["Real", "Predicha"]
    )
)

diagonal = (
    alt.Chart(scatter_df)
    .mark_line(color="red")
    .encode(x="Real:Q", y="Real:Q")
)

chart = (points + diagonal).properties(height=350)

st.altair_chart(chart, width="stretch")

# 4.6 Comparación (solo en modo avanzado)
if modo_experto:
    st.subheader("📊 Comparación completa de modelos")
    st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}))

# ==========================================================
# 5. SEMÁFORO OPERATIVO
# ==========================================================
st.markdown("## 🚦 Semáforo de riesgo por filtro")

best_model = models[best]
X_scaled_full = scaler.transform(X)
rul_pred_full = best_model.predict(X_scaled_full)

df_risk = df_fe.copy()
df_risk["RUL_pred"] = rul_pred_full

# Tomar el último estado por filtro
if "Data_No" in df_risk.columns and "Time" in df_risk.columns:
    df_risk_sorted = df_risk.sort_values(["Data_No", "Time"])
    idx_last = df_risk_sorted.groupby("Data_No")["Time"].idxmax()
    df_current = df_risk_sorted.loc[idx_last].copy()
else:
    st.error("No existe columna 'Data_No' para identificar cada filtro.")
    st.stop()

# Porcentaje de vida útil
max_rul_real = df["RUL"].max()
df_current["RUL_pct"] = df_current["RUL_pred"] / max_rul_real * 100

# Clasificación UX (umbral rojo configurable desde sidebar)
def clasificar_estado(pct):
    if pct < umbral_rojo:
        return "Rojo"
    elif pct < umbral_amarillo:
        return "Amarillo"
    else:
        return "Verde"

df_current["Estado"] = df_current["RUL_pct"].apply(clasificar_estado)

# KPIs
n_total = len(df_current)
n_rojo = (df_current["Estado"] == "Rojo").sum()
n_amar = (df_current["Estado"] == "Amarillo").sum()
n_verde = (df_current["Estado"] == "Verde").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total filtros", n_total)
c2.metric("🟢 Verdes", n_verde)
c3.metric("🟡 Amarillos", n_amar)
c4.metric("🔴 Rojos", n_rojo)

# Listado de rojos
st.markdown("### 🔴 Filtros que requieren intervención")

df_rojo = df_current[df_current["Estado"] == "Rojo"]

if df_rojo.empty:
    st.success("No hay filtros en estado rojo.")
else:
    st.dataframe(
        df_rojo[["Data_No", "RUL_pred", "RUL_pct", "Estado"]].sort_values("RUL_pred")
        .style.format({"RUL_pred": "{:.1f}", "RUL_pct": "{:.1f}"})
    )

# Tabla completa
with st.expander("📄 Ver tabla completa del estado actual de todos los filtros"):
    st.dataframe(
        df_current[["Data_No", "RUL_pred", "RUL_pct", "Estado"]]
        .sort_values("RUL_pct")
        .style.format({"RUL_pred": "{:.1f}", "RUL_pct": "{:.1f}"})
    )
