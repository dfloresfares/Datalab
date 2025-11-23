import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# 1. FEATURE ENGINEERING
# ==========================================================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Data_No" in df.columns and "Time" in df.columns:
        df = df.sort_values(["Data_No", "Time"])
    elif "Time" in df.columns:
        df = df.sort_values(["Time"])

    df["dp_diff"] = 0.0
    df["dp_slope"] = 0.0
    df["dp_rolling_mean"] = 0.0
    df["dp_rolling_std"] = 0.0

    window = 5

    if "Data_No" in df.columns:
        groups = df.groupby("Data_No")
    else:
        groups = [(None, df)]

    for key, g in groups:
        g = g.sort_values("Time") if "Time" in g.columns else g

        dp_diff = g["Differential_pressure"].diff().fillna(0.0)

        if "Time" in g.columns:
            time_diff = g["Time"].diff().replace(0, np.nan)
            dp_slope = (dp_diff / time_diff).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            dp_slope = dp_diff * 0.0

        dp_roll_mean = g["Differential_pressure"].rolling(window=window, min_periods=1).mean()
        dp_roll_std = g["Differential_pressure"].rolling(window=window, min_periods=2).std().fillna(0.0)

        df.loc[g.index, "dp_diff"] = dp_diff.values
        df.loc[g.index, "dp_slope"] = dp_slope.values
        df.loc[g.index, "dp_rolling_mean"] = dp_roll_mean.values
        df.loc[g.index, "dp_rolling_std"] = dp_roll_std.values

    eps = 1e-3
    df["dp_over_flow"] = df["Differential_pressure"] / (df["Flow_rate"] + eps)

    return df


# ==========================================================
# 2. PREPARAR FEATURES Y TARGET
# ==========================================================
def prepare_features(df: pd.DataFrame, target_col: str = "RUL"):
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

    num_cols = [c for c in num_cols if c in df_model.columns]
    dust_ohe_cols = [c for c in df_model.columns if c.startswith("Dust_")]

    feature_cols = num_cols + dust_ohe_cols

    X = df_model[feature_cols].values
    y = df_model[target_col].values

    return X, y, feature_cols


# ==========================================================
# 3. ENTRENAR Y EVALUAR MODELOS
# ==========================================================
def train_and_evaluate_models(X, y):
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

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Modelo": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        predictions[name] = {
            "y_test": y_test,
            "y_pred": y_pred
        }

    metrics_df = pd.DataFrame(results).sort_values(by="RMSE")
    return metrics_df, predictions


# ==========================================================
# 4. STREAMLIT APP
# ==========================================================
st.set_page_config(page_title="Comparación de modelos de vida útil remanente", page_icon="⚙️")

st.title("⚙️ Comparación de modelos para estimar vida útil remanente (VUR)")
st.write("Subí un dataset con columna `RUL` y evaluá automáticamente distintos modelos supervisados.")

uploaded_file = st.file_uploader("📂 Subí tu archivo CSV con datos del filtro", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())

    if "RUL" not in df.columns:
        st.error("El dataset debe contener la columna 'RUL'.")
        st.stop()

    st.subheader("Generando features derivados…")
    df_fe = add_derived_features(df)

    X, y, feature_cols = prepare_features(df_fe, target_col="RUL")

    st.info(f"Total de features usados: {len(feature_cols)}")

    st.subheader("Entrenando modelos…")
    metrics_df, predictions = train_and_evaluate_models(X, y)

    st.subheader("📊 Resultados comparativos")
    st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}))

    best_model = metrics_df.iloc[0]["Modelo"]
    st.success(f"Mejor modelo según RMSE: **{best_model}**")

    st.subheader("🔍 Vida útil real vs predicha (scatter)")
    y_test = predictions[best_model]["y_test"]
    y_pred = predictions[best_model]["y_pred"]

    corr = np.corrcoef(y_test, y_pred)[0, 1]
    st.write(f"Correlación: **{corr:.3f}**")

    scatter_df = pd.DataFrame({"Vida real": y_test, "Vida predicha": y_pred})
    st.scatter_chart(scatter_df, x="Vida real", y="Vida predicha")

else:
    st.info("Subí un archivo CSV para comenzar.")

