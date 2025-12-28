import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from sklearn.model_selection import train_test_split, KFold

from bsm_model import BlackScholesModel
from xgb_model import XGBoostModel, EvaluationMetrics

# Helpers for underlying history data
def realized_vol(returns, window=20):
    """Rolling realized volatility (annualized)."""
    return returns.rolling(window=window).std() * np.sqrt(252)


def fetch_tsla_history(start="2020-01-01"):
    """Download TSLA daily history and compute realized vols."""
    df = yf.download("TSLA", start=start, auto_adjust=False, progress=False)

    # Normalize columns in case of multiindex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    px = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["Return"] = df[px].pct_change()
    df["RealizedVol20"] = realized_vol(df["Return"], 20)
    df["RealizedVol60"] = realized_vol(df["Return"], 60)
    df.dropna(inplace=True)
    return df

def load_bloomberg_chain(path):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Basic sanity checks
    required = ["Expiry", "Put/Call Strike", "Call Bid", "Call Ask", "Put Bid", "Put Ask"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in chain CSV: {missing}")

    # Parse expiry as datetime
    df["Expiry"] = pd.to_datetime(df["Expiry"])

    # Calls
    calls = df[["Expiry", "Put/Call Strike", "Call Bid", "Call Ask"]].copy()
    calls.rename(columns={
        "Expiry": "expiry",
        "Put/Call Strike": "strike",
        "Call Bid": "bid",
        "Call Ask": "ask",
    }, inplace=True)
    calls["option_type"] = "call"

    # Puts
    puts = df[["Expiry", "Put/Call Strike", "Put Bid", "Put Ask"]].copy()
    puts.rename(columns={
        "Expiry": "expiry",
        "Put/Call Strike": "strike",
        "Put Bid": "bid",
        "Put Ask": "ask",
    }, inplace=True)
    puts["option_type"] = "put"

    options = pd.concat([calls, puts], ignore_index=True)
    return options


def load_vol_surface(path):
    surf = pd.read_csv(path)
    surf.columns = [c.strip() for c in surf.columns]

    bucket_cols = [c for c in surf.columns if "%" in c]
    if len(bucket_cols) == 0:
        raise ValueError("No moneyness bucket columns (e.g. '80.0%') found in surface CSV.")

    # Strikes for each bucket
    strike_row = surf.iloc[0]
    strike_buckets = strike_row[bucket_cols].astype(float).values

    # Implied vols (percent)
    iv_row = surf.iloc[1]
    iv_buckets = iv_row[bucket_cols].astype(float).values / 100.0

    return {"bucket_cols": bucket_cols, "strike_buckets": strike_buckets, "iv_buckets": iv_buckets }


def approximate_iv_from_surface(strikes, surface_info):
    K = np.asarray(strikes, dtype=float)
    K_b = surface_info["strike_buckets"]
    iv_b = surface_info["iv_buckets"]

    # Simple 1D interpolation, clipping outside the bucket range
    iv = np.interp(K, K_b, iv_b, left=iv_b[0], right=iv_b[-1])
    return iv

# Data Preparation and Preprocessing
def build_tsla_option_dataset(chain_csv, surface_csv):
    # Underlying history
    hist = fetch_tsla_history()
    last_date = hist.index[-1]

    px = "Adj Close" if "Adj Close" in hist.columns else "Close"
    S = hist.loc[last_date, px]
    rv20 = hist.loc[last_date, "RealizedVol20"]
    rv60 = hist.loc[last_date, "RealizedVol60"]
    r = 0.01  # constant risk-free rate

    print(f"Using TSLA spot S={S:.2f} on {last_date.date()}")
    print(f"RealizedVol20={rv20:.4f}, RealizedVol60={rv60:.4f}")

    # Bloomberg chain
    opt = load_bloomberg_chain(chain_csv)
    opt["as_of"] = last_date

    # Mid-price and basic cleaning
    opt["mid"] = (opt["bid"] + opt["ask"]) / 2.0
    opt = opt[(opt["mid"] > 0) & (opt["ask"] >= opt["bid"]) & (opt["bid"] >= 0)]

    # Time to maturity in years
    opt["T"] = (opt["expiry"] - opt["as_of"]).dt.days / 365.0
    # drop options expiring same day
    opt = opt[opt["T"] > 1 / 365]   

    # Underlying and realized vols
    opt["S"] = S
    opt["RealizedVol20"] = rv20
    opt["RealizedVol60"] = rv60
    opt["type_flag"] = opt["option_type"].map({"call": 1, "put": 0})

    # Moneyness features based on spot
    opt["moneyness"] = opt["S"] / opt["strike"]
    opt["log_moneyness"] = np.log(opt["moneyness"])

    # Approximate implied vol from surface as an extra feature
    surface_info = load_vol_surface(surface_csv)
    opt["iv_from_surface"] = approximate_iv_from_surface(opt["strike"].values, surface_info)

    # Relative spread filter
    opt["rel_spread"] = (opt["ask"] - opt["bid"]) / opt["mid"]
    opt = opt[opt["rel_spread"] < 0.6]

    # Price sanity filters
    opt = opt[(opt["strike"] > 1) & (opt["strike"] < 2000)]
    opt = opt[opt["mid"] < 500]

    # BSM baseline using 20-day realized volatility as sigma
    opt["bs_price"] = BlackScholesModel.bsm_price(
        S=opt["S"].values,
        K=opt["strike"].values,
        T=opt["T"].values,
        r=r,
        sigma=opt["RealizedVol20"].values,
        option_type=opt["option_type"].values,
    )

    opt = opt[opt["bs_price"] > 0]
    opt.dropna(inplace=True)
    print(f"Number of TSLA option contracts after cleaning: {len(opt)}")
    return opt


def prepare_features_and_residual_target(df):
    feature_names = [
        "S",
        "strike",
        "T",
        "moneyness",
        "log_moneyness",
        "RealizedVol20",
        "RealizedVol60",
        "type_flag",
        "bs_price",
        "iv_from_surface",
    ]

    X = df[feature_names].fillna(0.0)
    y_mid = df["mid"].values
    bs = df["bs_price"].values
    y_resid = y_mid - bs
    return X, y_resid, y_mid, bs, feature_names

# Plot utilities
def ensure_fig_dir():
    if not os.path.exists("figures"):
        os.makedirs("figures")


def scatter(true_mid, bs_pred, xgb_pred, fname):
    ensure_fig_dir()
    plt.figure(figsize=(8, 5))
    plt.scatter(true_mid, bs_pred, s=18, alpha=0.7, label="BSM")
    plt.scatter(true_mid, xgb_pred, s=18, alpha=0.7, label="XGBoost")
    max_p = max(true_mid.max(), bs_pred.max(), xgb_pred.max())
    plt.plot([0, max_p], [0, max_p], "k--", linewidth=1)
    plt.xlabel("Actual mid-price")
    plt.ylabel("Predicted mid-price")
    plt.title("TSLA: Actual vs Predicted Mid-Prices (Test Set)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join("figures", fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def rmse_by_moneyness(df_test, bs_pred, xgb_pred, fname):
    ensure_fig_dir()
    m = df_test["S"] / df_test["strike"]
    buckets = pd.cut(m, bins=[0, 0.8, 1.2, 100], labels=["Deep OTM", "Near ATM", "Deep ITM"])

    true = df_test["mid"].values
    rmse_bs, rmse_xgb, labels = [], [], []

    for label in buckets.cat.categories:
        mask = buckets == label
        if mask.sum() == 0:
            continue
        labels.append(str(label))
        rmse_bs.append(EvaluationMetrics.root_mean_squared_error(true[mask], bs_pred[mask]))
        rmse_xgb.append(EvaluationMetrics.root_mean_squared_error(true[mask], xgb_pred[mask]))

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - w / 2, rmse_bs, w, label="BSM")
    plt.bar(x + w / 2, rmse_xgb, w, label="XGBoost")
    plt.xticks(x, labels)
    plt.ylabel("RMSE")
    plt.title("TSLA: RMSE by Moneyness Bucket (Test Set)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join("figures", fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def feature_importance(model, feature_names, fname):
    ensure_fig_dir()
    imp = model.model.feature_importances_
    idx = np.argsort(imp)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp = imp[idx]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(sorted_names)), sorted_imp)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("TSLA: XGBoost Feature Importance")
    plt.tight_layout()
    out = os.path.join("figures", fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def cv_boxplot(rmse_bs, rmse_xgb, fname):
    ensure_fig_dir()
    data = [rmse_bs, rmse_xgb]
    labels = ["BSM", "XGBoost"]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels)
    plt.scatter([1, 2], [np.mean(rmse_bs), np.mean(rmse_xgb)],
                marker="^")
    plt.ylabel("RMSE across folds")
    plt.title("TSLA: Cross-validated RMSE")
    plt.tight_layout()
    out = os.path.join("figures", fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def error_heatmaps(df, bs_pred, xgb_pred, prefix):
    ensure_fig_dir()
    true = df["mid"].values
    m = df["S"] / df["strike"]
    T = df["T"].values

    m_bins = np.array([0.5, 0.8, 1.0, 1.2, 2.0])
    T_bins = np.array([0.0, 0.1, 0.25, 0.5, 1.0])

    m_labels = [f"{m_bins[i]:.2f}-{m_bins[i+1]:.2f}" for i in range(len(m_bins)-1)]
    T_labels = [f"{T_bins[i]:.2f}-{T_bins[i+1]:.2f}" for i in range(len(T_bins)-1)]

    def grid_mae(pred):
        g = np.zeros((len(T_bins) - 1, len(m_bins) - 1))
        g[:] = np.nan
        for i in range(len(T_bins) - 1):
            for j in range(len(m_bins) - 1):
                mask = (
                    (T >= T_bins[i]) & (T < T_bins[i + 1]) &
                    (m >= m_bins[j]) & (m < m_bins[j + 1])
                )
                if mask.sum() > 0:
                    g[i, j] = EvaluationMetrics.mean_absolute_error(true[mask], pred[mask])
        return g

    mae_bs = grid_mae(bs_pred)
    mae_xgb = grid_mae(xgb_pred)

    vmin = np.nanmin([mae_bs, mae_xgb])
    vmax = np.nanmax([mae_bs, mae_xgb])

    for grid, title, suf in [
        (mae_bs, "BSM mean absolute error", "bsm"),
        (mae_xgb, "XGBoost mean absolute error", "xgb"),
    ]:
        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            grid,
            origin="lower",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        plt.colorbar(im, label="MAE")
        x_ticks = np.arange(len(m_labels))
        y_ticks = np.arange(len(T_labels))
        plt.xticks(x_ticks, m_labels, rotation=90)
        plt.yticks(y_ticks, T_labels)
        plt.xlabel("Moneyness S/K (bins)")
        plt.ylabel("Time to maturity T (years, bins)")
        plt.title(f"TSLA: {title}")
        plt.tight_layout()
        out = os.path.join("figures", f"{prefix}_{suf}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Saved {out}")


def error_hist(true_mid, bs_pred, xgb_pred, fname):
    ensure_fig_dir()
    err_bs = np.abs(true_mid - bs_pred)
    err_xgb = np.abs(true_mid - xgb_pred)

    plt.figure(figsize=(8, 5))
    plt.hist(err_bs, bins=40, alpha=0.6, label="BSM")
    plt.hist(err_xgb, bins=40, alpha=0.6, label="XGBoost")
    plt.xlabel("Absolute error |y - ŷ|")
    plt.ylabel("Count")
    plt.title("TSLA: Distribution of absolute pricing errors")
    plt.legend()
    plt.tight_layout()
    out = os.path.join("figures", fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def error_cdf(true_mid, bs_pred, xgb_pred, fname):
    ensure_fig_dir()
    err_bs = np.abs(true_mid - bs_pred)
    err_xgb = np.abs(true_mid - xgb_pred)

    err_bs_sorted = np.sort(err_bs)
    err_xgb_sorted = np.sort(err_xgb)
    n_bs = len(err_bs_sorted)
    n_xgb = len(err_xgb_sorted)
    cdf_bs = np.arange(1, n_bs + 1) / n_bs
    cdf_xgb = np.arange(1, n_xgb + 1) / n_xgb

    plt.figure(figsize=(8, 5))
    plt.plot(err_bs_sorted, cdf_bs, label="BSM")
    plt.plot(err_xgb_sorted, cdf_xgb, label="XGBoost")
    plt.xlabel("Absolute error |y - ŷ|")
    plt.ylabel("Cumulative fraction")
    plt.title("TSLA: CDF of absolute pricing errors")
    plt.legend()
    plt.tight_layout()
    out = os.path.join("figures", fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def relative_improvement(true_mid, bs_pred, xgb_pred, fname_prefix):
    ensure_fig_dir()
    err_bs = np.abs(true_mid - bs_pred)
    err_xgb = np.abs(true_mid - xgb_pred)
    rel_improvement = (err_bs - err_xgb) / (err_bs + 1e-8)

    plt.figure(figsize=(8, 5))
    plt.scatter(true_mid, rel_improvement, s=15, alpha=0.6)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Actual mid-price")
    plt.ylabel("Relative improvement")
    plt.title("TSLA: Relative improvement of XGBoost over BSM")
    plt.tight_layout()
    out = os.path.join("figures", f"{fname_prefix}_rel_improvement_scatter.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

    plt.figure(figsize=(8, 5))
    plt.hist(rel_improvement, bins=40, alpha=0.8)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Relative improvement (BSM error - XGB error) / BSM error")
    plt.ylabel("Count")
    plt.title("TSLA: Distribution of XGBoost relative improvement")
    plt.tight_layout()
    out = os.path.join("figures", f"{fname_prefix}_rel_improvement_hist.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def correlation_heatmap(X, fname):
    ensure_fig_dir()
    corr = X.corr()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        corr.values,
        origin="lower",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(im, label="Correlation")

    ticks = np.arange(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=90)
    plt.yticks(ticks, corr.columns)

    plt.title("Correlation heatmap of features")
    plt.tight_layout()
    out = os.path.join("figures", fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def validation_curve(X_train, X_test, y_resid_train, y_resid_test, bs_train, bs_test, true_mid_train, true_mid_test, base_params, fname):
    ensure_fig_dir()

    n_estimators_grid = [50, 100, 200, 400, 600]
    rmse_train = []
    rmse_test = []

    for n_estimators in n_estimators_grid:
        params = base_params.copy()
        params["n_estimators"] = n_estimators

        model = XGBoostModel(**params)
        model.fit(X_train, y_resid_train)

        resid_train_pred = model.predict(X_train)
        resid_test_pred = model.predict(X_test)

        xgb_train_pred = bs_train + resid_train_pred
        xgb_test_pred = bs_test + resid_test_pred

        rmse_train.append(EvaluationMetrics.root_mean_squared_error(true_mid_train, xgb_train_pred))
        rmse_test.append(EvaluationMetrics.root_mean_squared_error(true_mid_test, xgb_test_pred))

    rmse_bs = EvaluationMetrics.root_mean_squared_error(true_mid_test, bs_test)

    plt.figure(figsize=(8, 5))
    plt.plot(n_estimators_grid, rmse_train, marker="o", label="XGBoost train RMSE")
    plt.plot(n_estimators_grid, rmse_test, marker="o", label="XGBoost test RMSE")
    plt.axhline(rmse_bs, linestyle="--", label="BSM test RMSE")
    plt.xlabel("Number of trees (n_estimators)")
    plt.ylabel("RMSE (mid-price)")
    plt.title("Validation curve: XGBoost complexity vs RMSE")
    plt.legend()
    plt.tight_layout()
    out = os.path.join("figures", fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


# Experiment trial
def experiment_single_split(chain_csv, surface_csv):
    df = build_tsla_option_dataset(chain_csv, surface_csv)
    X, y_resid, y_mid, bs_all, feature_names = prepare_features_and_residual_target(df)
    correlation_heatmap(X, "fig_feature_correlation_tsla.png")

    X_train, X_test, y_resid_train, y_resid_test = train_test_split(X, y_resid, test_size=0.2, random_state=42)

    true_mid_train = df.loc[X_train.index, "mid"].values
    true_mid_test = df.loc[X_test.index, "mid"].values
    bs_train = df.loc[X_train.index, "bs_price"].values
    bs_test = df.loc[X_test.index, "bs_price"].values

    base_params = dict(
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=5,
        gamma=0.0,
        n_estimators=200,
    )

    model = XGBoostModel(**base_params)
    model.fit(X_train, y_resid_train)

    resid_pred_test = model.predict(X_test)
    xgb_pred_test = bs_test + resid_pred_test

    eval_bs = EvaluationMetrics.evaluate_model(true_mid_test, bs_test)
    eval_xgb = EvaluationMetrics.evaluate_model(true_mid_test, xgb_pred_test)

    print("\n=== Single train/test split (residual-correction XGBoost) ===")
    print(f"BSM    : RMSE={eval_bs['RMSE']:.4f}, R^2={eval_bs['R^2']:.4f}")
    print(f"XGBoost: RMSE={eval_xgb['RMSE']:.4f}, R^2={eval_xgb['R^2']:.4f}")
    print(f"ΔRMSE (XGB - BSM) = {eval_xgb['RMSE'] - eval_bs['RMSE']:.4f}")
    print(f"ΔR^2  (XGB - BSM) = {eval_xgb['R^2'] - eval_bs['R^2']:.6f}")

    scatter(true_mid_test, bs_test, xgb_pred_test, "fig_prices_scatter_tsla.png")
    rmse_by_moneyness(df.loc[X_test.index], bs_test, xgb_pred_test, "fig_rmse_by_moneyness_tsla.png")
    feature_importance(model, feature_names, "fig_feature_importance_tsla.png")
    error_heatmaps(df.loc[X_test.index], bs_test, xgb_pred_test, "fig_heatmap_error_tsla")
    error_hist(true_mid_test, bs_test, xgb_pred_test, "fig_error_hist_tsla.png")
    error_cdf(true_mid_test, bs_test, xgb_pred_test, "fig_error_cdf_tsla.png")
    relative_improvement(true_mid_test, bs_test, xgb_pred_test, "fig_tsla")

    validation_curve(
        X_train, X_test,
        y_resid_train, y_resid_test,
        bs_train, bs_test,
        true_mid_train, true_mid_test,
        base_params,
        "fig_validation_curve_tsla.png",
    )

    return df, X, y_mid


def experiment_crossval(df, X, y_mid, n_splits=3):
    X_full, y_resid, _, bs_all, _ = prepare_features_and_residual_target(df)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_bs, rmse_xgb, r2_bs, r2_xgb = [], [], [], []

    params = dict(
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=5,
        gamma=0.0,
        n_estimators=200,
    )

    print(f"\n=== Cross-validated backtest (residual XGB, K={n_splits}) ===")
    for fold, (tr, te) in enumerate(kf.split(X_full), start=1):
        print(f"  Fold {fold}/{n_splits}")
        X_train, X_test = X_full.iloc[tr], X_full.iloc[te]
        y_resid_train, y_resid_test = y_resid[tr], y_resid[te]

        true_mid = y_mid[te]
        bs_test = bs_all[te]

        model = XGBoostModel(**params)
        model.fit(X_train, y_resid_train)
        resid_pred = model.predict(X_test)
        xgb_pred = bs_test + resid_pred

        ev_bs = EvaluationMetrics.evaluate_model(true_mid, bs_test)
        ev_xgb = EvaluationMetrics.evaluate_model(true_mid, xgb_pred)

        rmse_bs.append(ev_bs["RMSE"])
        rmse_xgb.append(ev_xgb["RMSE"])
        r2_bs.append(ev_bs["R^2"])
        r2_xgb.append(ev_xgb["R^2"])

    print("\n=== Cross-validated summary ===")
    print(f"BSM   RMSE mean={np.mean(rmse_bs):.4f}")
    print(f"XGB   RMSE mean={np.mean(rmse_xgb):.4f}")
    print(f"BSM    R^2 mean={np.mean(r2_bs):.4f}")
    print(f"XGB    R^2 mean={np.mean(r2_xgb):.4f}")
    print(f"ΔRMSE mean (XGB - BSM) = {np.mean(rmse_xgb) - np.mean(rmse_bs):.4f}")
    print(f"ΔR^2  mean (XGB - BSM) = {np.mean(r2_xgb) - np.mean(r2_bs):.6f}")

    cv_boxplot(rmse_bs, rmse_xgb, "fig_cv_rmse_boxplot_tsla.png")


# Main
if __name__ == "__main__":
    chain_csv = "options_chain_12dec2025.csv"
    surface_csv = "tesla_volatility_surface.csv"

    df, X, y_mid = experiment_single_split(chain_csv, surface_csv)
    experiment_crossval(df, X, y_mid, n_splits=3)
