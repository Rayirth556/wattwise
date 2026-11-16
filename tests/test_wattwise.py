import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import joblib, numpy as np, pandas as pd
import utils as U


def synth(n=60, start="2010-01-01 00:00"):
    idx = pd.date_range(start, periods=n, freq="min")
    return pd.DataFrame({"Global_active_power": np.linspace(0.5, 2, n),
                         "Global_reactive_power": 0.1, "Voltage": 240.0, "Global_intensity": 2.0,
                         "Sub_metering_1": 0., "Sub_metering_2": 0., "Sub_metering_3": 0.},
                        index=idx).astype("float32")


def test_rolling_uses_past_only():
    df = synth(); a = U.add_features(df)
    df2 = df.copy(); df2.iloc[40:, 0] = 9.0      # change the future
    b = U.add_features(df2)
    pd.testing.assert_series_equal(a["roll_mean_15"].iloc[:40], b["roll_mean_15"].iloc[:40])


def test_features_and_missing():
    df = synth(); df.iloc[20, 0] = np.nan
    f = U.add_features(df)
    assert not f["valid"].iloc[20] and f["valid"].iloc[30]
    assert f["is_weekend"].iloc[0] == 0 and f["hour"].iloc[0] == 0   # Fri 2010-01-01


def test_labels():
    f = U.add_features(synth())
    th = {"high_power_kw": 1.5, "standby_power_kw": 0.6, "standby_roll_std_kw": 0.5}
    lab = U.assign_labels(f[f["valid"]], th)
    assert set(lab) <= set(U.CLASSES) and lab.iloc[-1] == "Unusually High"


def test_energy():
    p = pd.Series(2.0, index=pd.date_range("2010-01-01", periods=60, freq="min"))
    assert abs(U.energy_kwh(p)["kwh"] - 2.0) < 1e-9          # 2 kW for 1 h = 2 kWh
    p.iloc[10:20] = np.nan                                      # gap: excluded, reported
    r = U.energy_kwh(p)
    assert r["n_missing"] == 10 and abs(r["kwh"] - 50 * 2 / 60) < 1e-9


def test_model_predictions_valid():
    model = joblib.load(U.ARTIFACT_DIR / "wattwise_rf.joblib")
    f = U.add_features(synth()); f = f[f["valid"]]
    assert set(U.predict(model, f)) <= set(U.CLASSES)
