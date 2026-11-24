from pathlib import Path
import io
import json
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


import plotly.graph_objects as go

AMBER = "#D99A55"
AMBER_LIGHT = "#F0BD83"
AMBER_DARK = "#A96D32"
CHART_BG = "#1C211E"
GRID_COLOR = "#343B35"
TEXT_COLOR = "#E8EAE6"

VANTAGE_PALETTE = [
    "#D99A55",  # Signature amber
    "#F0BD83",  # Champagne amber
    "#B97835",  # Deep copper
    "#E8A65B",  # Golden amber
    "#C98A45",  # Bronze
    "#F4D2A1",  # Pale gold
]

def style_vantage_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CHART_BG,
        font=dict(family="Sora, sans-serif", color=TEXT_COLOR),
        title_font=dict(
            family="Sora, sans-serif",
            color=AMBER_LIGHT
        ),
        colorway=[
            "#D99A55",  # Primary amber
            "#E8BD83",  # Champagne
            "#A96D32",  # Copper
            "#F0D6AE",  # Pale gold
            "#BD8750",  # Muted bronze
            "#8D775D"   # Dark bronze
        ],
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR
        ),
        legend=dict(font=dict(color=TEXT_COLOR)),
        margin=dict(l=45, r=25, t=55, b=40)
    )

    # Force distinct VANTAGE amber shades across all traces
    for i, trace in enumerate(fig.data):
        color = VANTAGE_PALETTE[i % len(VANTAGE_PALETTE)]

        if trace.type == "scatter":
            if trace.mode and "lines" in trace.mode:
                trace.line.color = color
                trace.line.width = 2.5

            if trace.mode and "markers" in trace.mode:
                trace.marker.color = color
                trace.marker.line.color = color

        elif trace.type in ("bar", "histogram"):
            trace.marker.color = color

    # Use amber for reference lines and chart shapes
    fig.update_shapes(line_color=AMBER)

    return fig


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Michroma&display=swap');

.vantage-wordmark {
    font-family: 'Michroma', sans-serif;
    font-size: 2.5rem;
    font-weight: 400;
    letter-spacing: 5px;
    color: #D99A55;
    line-height: 1.4;
    margin-bottom: 4px;
}

.vantage-tagline {
    font-family: 'Sora', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 3px;
    color: #929992;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data_utils import COLUMNS, SENSORS, SETTINGS, load_subset, make_features, label_test

DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

st.set_page_config(page_title="VANTAGE | Engine Health", page_icon="âœˆï¸", layout="wide")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_recorded_subset(subset):
    _, test, rul = load_subset(DATA_DIR, subset)
    labelled = label_test(test, rul)
    return test, labelled, rul

@st.cache_data
def load_reports():
    metrics_path = REPORT_DIR / "benchmark_results.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    robustness_path = REPORT_DIR / "robustness_results.json"
    robustness = json.loads(robustness_path.read_text()) if robustness_path.exists() else {}
    noise_path = REPORT_DIR / "noise_augmentation_results.json"
    noise = json.loads(noise_path.read_text()) if noise_path.exists() else {}
    return metrics, robustness, noise



def get_available_models(subset):
    found = {}

    # Original baseline models
    baseline_path = MODEL_DIR / f"xgboost_{subset}.joblib"
    ridge_path = MODEL_DIR / f"ridge_{subset}.joblib"

    # Tuned model
    tuned_path = MODEL_DIR / f"xgboost_tuned_{subset}.joblib"

    # Keep both models available for comparison
    if baseline_path.exists():
        found["XGBoost (baseline)"] = baseline_path

    if tuned_path.exists():
        found["XGBoost (tuned)"] = tuned_path

    if ridge_path.exists():
        found["Ridge"] = ridge_path

    # Noise-augmented model
    noise_path = MODEL_DIR / f"xgboost_noiseaug_{subset}.joblib"
    if noise_path.exists():
        found["XGBoost (noise-augmented)"] = noise_path

    return found


def engineer_features(frame):
    features = make_features(frame, window=30)
    # Use the exact feature order stored in the fitted estimator.
    return features


def parse_upload(uploaded_file):
    raw = uploaded_file.getvalue()
    # Accept canonical CSV headers, headerless comma-separated files, and
    # original whitespace-separated NASA C-MAPSS rows.
    try:
        candidate = pd.read_csv(io.BytesIO(raw))
        normalized = [str(c).strip().lower() for c in candidate.columns]
        if set(COLUMNS).issubset(set(normalized)):
            candidate.columns = normalized
            frame = candidate[COLUMNS].copy()
        elif set(["cycle", *SETTINGS, *SENSORS]).issubset(set(normalized)):
            candidate.columns = normalized
            candidate.insert(0, "unit", 1)
            frame = candidate[COLUMNS].copy()
        else:
            raise ValueError("No canonical header detected")
    except Exception:
        first_line = raw.decode("utf-8-sig", errors="ignore").splitlines()[0] if raw else ""
        delimiter = "," if "," in first_line else r"\s+"
        candidate = pd.read_csv(io.BytesIO(raw), header=None, sep=delimiter, engine="python")
        if candidate.shape[1] == 26:
            candidate.columns = COLUMNS
            frame = candidate[COLUMNS].copy()
        elif candidate.shape[1] == 25:
            candidate.columns = ["cycle", *SETTINGS, *SENSORS]
            candidate.insert(0, "unit", 1)
            frame = candidate[COLUMNS].copy()
        else:
            raise ValueError(f"Expected 26 columns (unit, cycle, 3 settings, 21 sensors), or 25 columns without unit. Found {candidate.shape[1]} columns.")
    for col in COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[COLUMNS].isna().any().any():
        bad = frame[COLUMNS].columns[frame[COLUMNS].isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values found in: {', '.join(bad)}")
    if frame.empty:
        raise ValueError("The uploaded CSV is empty.")
    frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    if (frame["cycle"] <= 0).any():
        raise ValueError("Cycle values must be positive.")
    return frame


def predict_trajectories(frame, subset, model_label):
    paths = get_available_models(subset)
    if model_label not in paths:
        raise FileNotFoundError(f"Model '{model_label}' is not available for {subset}.")
    model = load_model(paths[model_label])
    features = engineer_features(frame)
    expected = list(getattr(model, "feature_names_in_", []))
    if expected:
        features = features[expected]
    pred_raw = np.asarray(model.predict(features), dtype=float)
    pred = np.clip(pred_raw, 0.0, 125.0)
    out = frame[["unit", "cycle"]].copy()
    out["predicted_RUL_raw"] = pred_raw
    out["predicted_RUL"] = pred
    out = out.sort_values(["unit", "cycle"])
    return out, model


def metric_card(label, value, suffix=""):
    st.metric(label, f"{value:,.1f}{suffix}" if pd.notna(value) else "â€”")



st.markdown("""
<div class="vantage-wordmark">VANTAGE</div>
<div class="vantage-tagline">Aircraft Intelligence System</div>
""", unsafe_allow_html=True)

st.caption("Aircraft engine health monitoring Â· Remaining Useful Life research dashboard")
st.warning("Research prototype using simulated NASA C-MAPSS data. Predictions are not certified for aircraft operation or maintenance decisions.")

with st.sidebar:
    st.header("Analysis settings")
    data_mode = st.radio("Data source", ["Recorded C-MAPSS engines", "Upload engine CSV"], index=0)
    subset = st.selectbox("C-MAPSS subset / model family", SUBSETS, help="Choose the model trained for the operating-condition and fault-mode subset that best matches the data.")
    available = get_available_models(subset)
    
    default_model = (
        "XGBoost (baseline)"
        if subset == "FD001"
        else "XGBoost (tuned)"
    )

    model_options = list(available.keys())

    default_index = (
        model_options.index(default_model)
        if default_model in model_options
        else 0
    )

    model_label = st.selectbox(
        "Prediction model",
        model_options,
        index=default_index if model_options else None
    )

    st.divider()
    st.caption("Models were trained with a 30-cycle causal rolling window and 67 engineered features.")

metrics_df, robustness, noise_results = load_reports()

if data_mode == "Recorded C-MAPSS engines":
    try:
        raw_test, labelled_test, rul_vector = load_recorded_subset(subset)
    except Exception as exc:
        st.error(f"Could not load the recorded dataset: {exc}")
        st.stop()
    engine_ids = sorted(raw_test.unit.unique().tolist())
    selected_unit = st.sidebar.selectbox("Engine ID", engine_ids)
    engine_frame = raw_test[raw_test.unit == selected_unit].copy().sort_values("cycle")
    known_rul = float(rul_vector[int(np.where(np.sort(engine_ids) == selected_unit)[0][0])])
    input_label = f"{subset} Â· test engine {selected_unit}"
    show_truth = True
else:
    uploaded = st.file_uploader("Upload a CSV containing one or more engine trajectories", type=["csv", "txt"], help="Required columns: unit, cycle, setting_1â€“3, sensor_1â€“21. A single-engine file may omit unit. Headerless 25/26-column NASA-style files are also accepted.")
    if uploaded is None:
        st.info("Upload a trajectory CSV to begin. The expected schema and example format are described in the Input guide tab below.")
        uploaded_frame = None
    else:
        try:
            uploaded_frame = parse_upload(uploaded)
            units = sorted(uploaded_frame.unit.unique().tolist())
            selected_unit = st.sidebar.selectbox("Uploaded engine ID", units)
            engine_frame = uploaded_frame[uploaded_frame.unit == selected_unit].copy().sort_values("cycle")
            input_label = f"Uploaded data Â· engine {selected_unit}"
            known_rul = None
            show_truth = False
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
            st.stop()

if data_mode == "Recorded C-MAPSS engines" or (data_mode == "Upload engine CSV" and uploaded is not None and 'uploaded_frame' in locals() and uploaded_frame is not None):
    try:
        predictions, fitted_model = predict_trajectories(engine_frame, subset, model_label)
        final_prediction_raw = float(predictions.iloc[-1].predicted_RUL_raw)
        final_prediction = float(predictions.iloc[-1].predicted_RUL)
        final_cycle = int(engine_frame.cycle.max())
        last_sensor = engine_frame.iloc[-1]
        st.subheader("Engine overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Predicted RUL (capped)", final_prediction, " cycles")
        with c2: metric_card("Current cycle", final_cycle)
        with c3: metric_card("Observed cycles", len(engine_frame))
        with c4:
            if show_truth:
                metric_card("Reference RUL", known_rul, " cycles")
            else:
                st.metric("Reference RUL", "Not provided")
        if show_truth:
            # Report the error from the uncapped model output, not the clipped
            # dashboard value, so the caption matches its "Raw model output" label.
            err = final_prediction_raw - known_rul
            st.caption(f"Raw model output: {final_prediction_raw:.1f} cycles Â· Raw prediction error against NASA reference: {err:+.1f} cycles. Displayed RUL is capped to the 0â€“125-cycle training target range; this is a dataset comparison, not a maintenance threshold.")

        tab_overview, tab_sensors, tab_compare, tab_error, tab_robust, tab_guide = st.tabs(["Overview", "Sensor monitoring", "Model performance", "Error analysis", "Robustness", "Input guide"])
        with tab_overview:
            left, right = st.columns([1.5, 1])
            with left:
                plot_data = predictions[["cycle", "predicted_RUL"]].copy()
                if show_truth:
                    # The model was trained on RUL labels capped at 125 cycles.
                    # Compare against the correspondingly capped NASA reference trajectory.
                    plot_data["NASA reference RUL (capped)"] = np.minimum(
                        known_rul + (final_cycle - plot_data["cycle"]), 125.0
                    )
                fig = px.line(
                    plot_data, x="cycle", y=[c for c in plot_data.columns if c != "cycle"],
                    markers=True, title="Predicted vs reference RUL across observed cycles",
                    labels={"value": "RUL (cycles)", "variable": "Trajectory"}
                )
                # Make the two trajectories visually distinct and keep the full
                # target range visible; a truncated axis exaggerates small errors.
                for trace in fig.data:
                    if trace.name == "predicted_RUL":
                        trace.name = "Predicted RUL"
                        trace.update(
                            line={"color": "#38bdf8", "width": 2.5, "dash": "solid"},
                            marker={"size": 5, "symbol": "circle"},
                        )
                    else:
                        trace.name = "NASA reference RUL (capped)"
                        trace.update(
                            line={"color": "#22c55e", "width": 2.5, "dash": "dash"},
                            marker={"size": 5, "symbol": "diamond"},
                        )
                fig.update_layout(
                    xaxis_title="Engine cycle",
                    yaxis_title="RUL (cycles)",
                    yaxis={"range": [0, 130], "dtick": 25, "rangemode": "tozero"},
                    hovermode="x unified",
                    legend_title="",
                )
                fig = style_vantage_chart(fig)
                st.plotly_chart(fig, use_container_width=True)
                if show_truth:
                    st.caption("The reference trajectory is derived from the supplied NASA test RUL and capped at 125 cycles to match the model's training target. The uncapped final reference value remains shown above.")
                st.caption("Predictions are clipped to the model's 0â€“125-cycle training target range for display. Raw model outputs remain available in the diagnostic details below.")
            with right:
                st.markdown("**Latest sensor snapshot**")
                snapshot = pd.DataFrame({"Sensor": SENSORS, "Latest value": [last_sensor[s] for s in SENSORS]})
                st.dataframe(snapshot, hide_index=True, use_container_width=True, height=430)
            st.markdown("**Prediction summary**")
            st.write(f"The selected {model_label} model estimates **{final_prediction:.1f} cycles** remaining at cycle {final_cycle}. The input contains {len(engine_frame)} observed cycles from {input_label}.")
            with st.expander("Prediction diagnostics"):
                st.write(f"Final raw model output: {final_prediction_raw:.3f} cycles")
                st.write(f"Final displayed prediction after 0â€“125 clipping: {final_prediction:.3f} cycles")
                st.write(f"Prediction range across this trajectory: {predictions.predicted_RUL.min():.1f}â€“{predictions.predicted_RUL.max():.1f} cycles")
                st.write("Clipping is a display/evaluation convention matching the capped training target; it does not mean the physical engine life is limited to 125 cycles.")
        with tab_sensors:
            chosen_sensors = st.multiselect("Sensors to plot", SENSORS, default=["sensor_2", "sensor_3", "sensor_4"])
            if chosen_sensors:
                sensor_long = engine_frame.melt(id_vars="cycle", value_vars=chosen_sensors, var_name="Sensor", value_name="Reading")
                fig = px.line(sensor_long, x="cycle", y="Reading", color="Sensor", title="Sensor trajectories")
                fig.update_layout(hovermode="x unified")
                fig = style_vantage_chart(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one sensor to display its trajectory.")
        with tab_compare:
            st.markdown("**Official C-MAPSS test benchmark**")
            if not metrics_df.empty:
                subset_metrics = metrics_df[metrics_df.Subset == subset].copy()
                st.dataframe(subset_metrics[["Model", "Test_MAE_raw_RUL", "Test_RMSE_raw_RUL", "Test_R2_raw_RUL"]].rename(columns={"Test_MAE_raw_RUL":"Test MAE", "Test_RMSE_raw_RUL":"Test RMSE", "Test_R2_raw_RUL":"Test RÂ²"}), hide_index=True, use_container_width=True)
                st.caption("These are aggregate official test-set metrics from the project reports; they are not recalculated from the single selected engine.")
                if show_truth:
                    st.markdown("**Selected engine: model comparison**")
                    compare_rows = []
                    for name, path in get_available_models(subset).items():
                        m = load_model(path)
                        feat = engineer_features(engine_frame)
                        expected = list(getattr(m, "feature_names_in_", []))
                        if expected: feat = feat[expected]
                        p = max(float(m.predict(feat.iloc[[-1]])[0]), 0.0)
                        compare_rows.append({"Model": name, "Predicted RUL": p, "Absolute error": abs(p-known_rul)})
                    st.dataframe(pd.DataFrame(compare_rows).sort_values("Absolute error"), hide_index=True, use_container_width=True)
            else:
                st.info("Benchmark report is not available in this project copy.")
        with tab_error:
            st.markdown("**Per-engine error analysis Â· official C-MAPSS test predictions**")
            pred_path = REPORT_DIR / f"{subset}_xgb_test_predictions.csv"
            if pred_path.exists():
                err_df = pd.read_csv(pred_path)
                # The stored benchmark prediction files contain one final prediction
                # per test engine from the baseline XGBoost model.
                err_df["Residual"] = err_df["predicted_RUL"] - err_df["true_RUL_raw"]
                err_df["Absolute error"] = err_df["Residual"].abs()
                err_df["Error direction"] = np.where(
                    err_df["Residual"] > 0, "Overestimation",
                    np.where(err_df["Residual"] < 0, "Underestimation", "Exact")
                )
                hit10 = (err_df["Absolute error"] <= 10).mean() * 100
                hit20 = (err_df["Absolute error"] <= 20).mean() * 100
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Test engines", f"{len(err_df)}")
                e2.metric("Mean absolute error", f"{err_df['Absolute error'].mean():.2f} cycles")
                e3.metric("Within Â±10 cycles", f"{hit10:.1f}%")
                e4.metric("Within Â±20 cycles", f"{hit20:.1f}%")
                st.caption(
                    "This analysis uses the saved baseline XGBoost test predictions, "
                    "not the currently selected model. 'Within Â±N cycles' is a tolerance-based "
                    "hit rate for regression, not classification accuracy."
                )

                left_err, right_err = st.columns(2)
                with left_err:
                    fig_resid = px.histogram(
                        err_df, x="Residual", nbins=24,
                        title="Distribution of prediction errors",
                        labels={"Residual": "Residual (predicted âˆ’ actual RUL)"},
                    )
                    fig_resid.add_vline(x=0, line_dash="dash", line_color="#22c55e")
                    fig_resid.update_layout(bargap=0.08)
                    fig = style_vantage_chart(fig_resid)
                    st.plotly_chart(fig_resid, use_container_width=True)
                with right_err:
                    fig_scatter = px.scatter(
                        err_df, x="true_RUL_raw", y="Residual",
                        hover_data=["unit", "predicted_RUL", "Absolute error"],
                        title="Residuals versus actual RUL",
                        labels={
                            "true_RUL_raw": "Actual RUL (cycles)",
                            "Residual": "Residual (predicted âˆ’ actual)",
                        },
                    )
                    fig_scatter.add_hline(y=0, line_dash="dash", line_color="#22c55e")
                    fig = style_vantage_chart(fig_scatter)
                    st.plotly_chart(fig_scatter, use_container_width=True)

                st.markdown("**Largest per-engine prediction errors**")
                worst = err_df.sort_values("Absolute error", ascending=False).head(10)
                st.dataframe(
                    worst[["unit", "true_RUL_raw", "predicted_RUL", "Residual", "Absolute error", "Error direction"]]
                    .rename(columns={
                        "unit": "Engine ID",
                        "true_RUL_raw": "Actual RUL",
                        "predicted_RUL": "Predicted RUL",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )
                selected_error_engine = err_df.loc[
                    err_df["unit"] == selected_unit
                ] if show_truth else pd.DataFrame()
                if not selected_error_engine.empty:
                    row = selected_error_engine.iloc[0]
                    st.info(
                        f"Selected engine {int(selected_unit)} in the saved baseline XGBoost benchmark: "
                        f"actual RUL {row['true_RUL_raw']:.1f}, predicted RUL {row['predicted_RUL']:.1f}, "
                        f"residual {row['Residual']:+.1f} cycles."
                    )
            else:
                st.info(
                    f"No saved per-engine prediction file was found for {subset}. "
                    "Run the benchmark prediction export before using this analysis."
                )

        with tab_robust:
            st.markdown("**Recorded synthetic sensor-noise sensitivity**")
            if subset in robustness:
                rows = []
                for level, values in robustness[subset].items():
                    rows.append({"Noise level": f"{float(level)*100:.0f}%", "MAE": values["MAE"]["mean"], "RMSE": values["RMSE"]["mean"], "RÂ²": values["R2"]["mean"]})
                rob_df = pd.DataFrame(rows)
                st.dataframe(rob_df, hide_index=True, use_container_width=True)
                fig = px.line(rob_df, x="Noise level", y="MAE", markers=True, title=f"Baseline XGBoost sensitivity Â· {subset}")
                fig = style_vantage_chart(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("The stored multi-seed noise sensitivity experiment covers FD001 and FD004 only.")
            if subset in noise_results:
                st.markdown("**Noise augmentation experiment**")
                st.caption("Available only for the subsets and splits included in the stored experiment. See the research report for full protocol and caveats.")
                try:
                    experiment = noise_results[subset]
                    split_name = "validation" if "validation" in experiment else next(iter(experiment))
                    block = experiment[split_name]
                    rows = []
                    for model_name, levels in block.items():
                        for level, values in levels.items():
                            rows.append({"Model": model_name, "Noise": f"{float(level)*100:.0f}%", "MAE": values["MAE"]["mean"]})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                except Exception:
                    st.caption("The augmentation report is present but has a different structure; inspect reports/noise_augmentation_results.json for full details.")
        with tab_guide:
            st.markdown("**CSV schema**")
            st.code(",".join(COLUMNS) + "\n" + "1,1,0.0023,0.0001,100.0," + ",".join(["500.0"]*21), language="csv")
            st.markdown("Use one row per engine cycle. Keep each engine's rows together in the same file. Values must be numeric; do not include a target RUL column in place of sensor columns. The app calculates trailing rolling features from the uploaded history, so upload the available history for each engineâ€”not only the final row.")
            st.markdown("For uploaded data, choose the matching C-MAPSS subset/model family. The model is trained on simulated C-MAPSS measurements; predictions on real aircraft telemetry or differently scaled sensors may be invalid.")
            st.download_button("Download CSV template", pd.DataFrame(columns=COLUMNS).to_csv(index=False), file_name="vantage_input_template.csv", mime="text/csv")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.exception(exc)

st.divider()
st.caption("VANTAGE Â· Academic research demonstrator Â· Model outputs are estimates, not certified prognostics.")

