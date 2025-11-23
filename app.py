import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Intentamos importar XGBoost (si no está instalado, lo ignoramos)
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ==========================================================
# Funciones auxiliares
# ==========================================================

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features derivados pensados para capturar degradación local:
    - dp_diff: cambio de presión diferencial
    - dp_slope: pendiente aproximada d(ΔP)/dt
    - dp_rolling_mean: media móvil de ΔP
    - dp_rolling_std: desvío móvil de ΔP
    - dp_over_flow: ΔP / Flow_rate
    """
    df = df.copy()
    # Si tenemos Data_No y Time, usamos la estructura temporal
    if "Data_No" in df.columns and "Time" in df.columns:
        df = df.sort_values(["Data_No", "Time"])
    elif "Time" in df.columns:
        df = df.sort_values("Time")

    df["dp_diff"] = 0.0
    df["dp_slope"] = 0.0
    df["dp_rolling_mean"] = 0.0
    df["dp_rolling_std"] = 0.0

    window = 5

    if "Data_No" in df.columns:
        group_cols = ["Data_No"]
    else:
        group_cols = [None]  # tratar todo junto

    if group_cols[0] is None:
        groups = [(None, df)]
    else:
        groups = df.groupby("Data_No")

    for key, g in groups:
        g = g.sort_values("Time") if "Time" in g.columns else g

        dp_diff = g["Differential_pressure"].diff().fillna(0.0)
        if "Time" in g.columns:
            time_diff = g["Time"].diff().replace(0, np.nan)
            dp_slope = (dp_diff / time_diff).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            dp_slope = dp_diff * 0.0

        dp_roll_mean = (
            g["Differential_pressure"]
            .rolling(window=window, min_periods=1)
            .mean()
        )
        dp_roll_std = (
            g["Differential_pressure"]
            .rolling(window=window, min_periods=2)
            .std()
            .fillna(0.0)
        )

        df.loc[g.index, "dp_diff"] = dp_diff.values
        df.loc[g.index, "dp_slope"] = dp_slope.values
        df.loc[g.index, "dp_rolling_mean"] = dp_roll_mean.values
        df.loc[g.index, "dp_rolling_std"] = dp_roll_std.values

    # ΔP / Flow_rate (evitar división por cero)
    eps = 1e-3
    df["dp_over_flow"] = df["Differential_pressure"] / (df["Flow_rate"] + eps)

    return df


def prepare_features(df: pd.DataFrame, target_col: str = "RUL"):
    """
    Prepara X e y:
    - hace get_dummies de Dust si existe
    - selecciona columnas numéricas + derivadas
    """
    df_model = df.copy()

    if "Dust" in df_model.columns:
        df_model = pd.get_dummies(df_model, columns=["Dust"], drop_first=True)

    num_cols = [
        "Differential_pressure",
        "Flow_rate",
        "Time",
        "Dust_feed",
        "dp_slope",
        "dp_rolling_mean",
        "dp_rolling_std",
        "dp_over_flow",
    ]

    # filtramos solo las columnas que existan
    num_cols = [c for c in num_cols if c in df_model.columns]

    dust_ohe_cols = [c for c in df_model.columns if c.startswith("Dust_")]

    feature_cols = num_cols + dust_ohe_cols

    X = df_model[feature_cols].values
    y = df_model[target_col].values

    return X, y, feature_cols


def train_and_evaluate_models(X, y, use_xgb=True):
    """
    Entrena todos los modelos y devuelve:
    - metrics_df: DataFrame con MAE, RMSE, R2 por modelo
    - results: dict con y_pred y modelo entrenado por nombre
    """
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

    if use_xgb and XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

    results = {}
    metrics_list = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        metrics_list.append({
            "Modelo": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_test": y_test,
            "X_test_scaled": X_test_scaled,
        }

    metrics_df = pd.DataFrame(metrics_list).sort_values(by="RMSE")
    return metrics_df, results, scaler
        

# ==========================================================
# App de Streamlit
# ==========================================================

st.set_page_config(
    page_title="Comparación de modelos de vida útil remanente",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Comparación de modelos para estimar vida útil remanente")
st.write(
    "Subí un dataset (por ejemplo el de filtros industriales de Kaggle) con la columna de vida útil remanente "
    "(por defecto `RUL`) y compará distintos modelos supervisados."
)

st.markdown("---")

# Opciones en la barra lateral
st.sidebar.header("Configuración")
use_xgb = st.sidebar.checkbox("Incluir XGBoost", value=True)
target_col = st.sidebar.text_input("Nombre de la columna target (vida útil)", value="RUL")

uploaded_file = st.file_uploader("📂 Subí tu CSV de datos (por ejemplo Test_Data_CSV.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset cargado. Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

    st.subheader("👀 Vista rápida del dataset")
    st.dataframe(df.head())

    if target_col not in df.columns:
        st.error(f"No se encontró la columna target '{target_col}' en el dataset.")
        st.stop()

    # Chequeo mínimo de columnas esperadas para features derivados
    missing_basic = [c for c in ["Differential_pressure", "Flow_rate", "Time", "Dust_feed"] if c not in df.columns]
    if missing_basic:
        st.warning(
            f"Faltan columnas básicas para los features derivados: {missing_basic}. "
            "La app asume el formato del dataset de filtros de Kaggle."
        )

    # Agregamos features derivados
    st.subheader("⚙️ Generando features derivados")
    df_fe = add_derived_features(df)
    st.write("Columns actuales:", df_fe.columns.tolist())

    # Preparamos X e y
    X, y, feature_cols = prepare_features(df_fe, target_col=target_col)
    st.write(f"Número de features usados: {len(feature_cols)}")

    # Entrenamos y evaluamos modelos
    st.subheader("🚀 Entrenando y evaluando modelos")
    if use_xgb and not XGB_AVAILABLE:
        st.warning("XGBoost no está instalado. Se ignorará este modelo.")

    metrics_df, results, scaler = train_and_evaluate_models(X, y, use_xgb=use_xgb)

    st.markdown("### 📊 Resultados comparativos (ordenados por RMSE)")
    st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}))

    best_model_name = metrics_df.iloc[0]["Modelo"]
    st.success(f"Mejor modelo según RMSE: **{best_model_name}**")

    # Scatter: real vs predicho para el mejor modelo
    st.markdown("### 🔍 Vida útil real vs predicha (mejor modelo)")

    best_res = results[best_model_name]
    y_test = best_res["y_test"]
    y_pred = best_res["y_pred"]

    corr = np.corrcoef(y_test, y_pred)[0, 1]
    st.write(f"Correlación entre vida real y predicción del mejor modelo: **{corr:.3f}**")

    scatter_df = pd.DataFrame({
        "Vida real": y_test,
        "Vida predicha": y_pred
    })

    st.scatter_chart(scatter_df, x="Vida real", y="Vida predicha")

else:
    st.info("Subí un archivo CSV para comenzar.")

