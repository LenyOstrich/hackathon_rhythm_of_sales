import requests
import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler


def add_russian_calendar_features(
    df: pd.DataFrame, dt_column: str = "dt"
) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[dt_column]):
        df[dt_column] = pd.to_datetime(df[dt_column])

    holidays = set()
    preholidays = set()

    for year in (2024, 2025):
        url = (
            "https://raw.githubusercontent.com/d10xa/holidays-calendar/"
            f"master/json/consultant{year}.json"
        )
        calendar = requests.get(url, timeout=10).json()

        holidays.update(pd.to_datetime(calendar["holidays"]))
        preholidays.update(pd.to_datetime(calendar["preholidays"]))

    df["is_preholiday"] = df[dt_column].isin(preholidays)

    df["is_non_working_day_rus"] = df[dt_column].isin(holidays) | (
        df[dt_column].dt.weekday >= 5
    )

    return df


def add_date_features(df: pd.DataFrame, dt_column: str = "dt") -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[dt_column]):
        df[dt_column] = pd.to_datetime(df[dt_column])

    df["day_of_week"] = df[dt_column].dt.day_of_week
    df["day"] = df[dt_column].dt.day
    df["month"] = df[dt_column].dt.month
    df["week_of_year"] = df[dt_column].dt.isocalendar().week
    df["is_weeked"] = df["day_of_week"].isin([5, 6])

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "qty",
    lags: list = [1, 2, 3, 7, 14, 21, 30],
):
    """Добавляет лаговые признаки к DataFrame.

    Parameters:
    df: DataFrame с колонками ['nm_id', 'dt', target_col]
    target_col: название колонки с целевой переменной (продажи)
    lags: список лагов для создания

    Returns:
    DataFrame с добавленными лаговыми признаками
    """
    df = df.copy()
    df = df.sort_values(["nm_id", "dt"]).reset_index(drop=True)

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("nm_id")[target_col].shift(lag)

    windows = [3, 7, 14, 30]
    for window in windows:
        df[f"rolling_mean_{window}"] = df.groupby("nm_id")[
            target_col
        ].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = df.groupby("nm_id")[
            target_col
        ].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )

    for window in [7, 14]:
        df[f"rolling_max_{window}"] = df.groupby("nm_id")[
            target_col
        ].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
        )
        df[f"rolling_min_{window}"] = df.groupby("nm_id")[
            target_col
        ].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
        )

    df["ewm_alpha_0.3"] = df.groupby("nm_id")[target_col].transform(
        lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean()
    )

    df["same_day_last_week"] = df.groupby("nm_id")[target_col].shift(7)

    df["trend_7"] = df.groupby("nm_id")[target_col].transform(
        lambda x: x.shift(1)
        .rolling(window=7, min_periods=1)
        .apply(
            lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) > 1 else 0
        )
    )

    fill_lag_features(df)

    return df


def fill_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = [
        c
        for c in df.columns
        if any(x in c for x in ["lag", "rolling", "ewm", "same_day", "trend"])
    ]
    for col in lag_cols:
        df[f"{col}_isna"] = df[col].isna().astype(int)
    df[lag_cols] = df[lag_cols].fillna(0)
    return df


def add_item_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("nm_id")
        .agg(
            item_mean_qty=("qty", "mean"),
            item_std_qty=("qty", "std"),
            item_nonzero_ratio=("qty", lambda x: (x > 0).mean()),
            item_mean_price=("price", "mean"),
        )
        .fillna(0)
    )
    return df.merge(stats, on="nm_id", how="left")


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["nm_id", "dt"])

    df["lag_price_1"] = df.groupby("nm_id")["price"].shift(1)

    df["rolling_mean_price_7"] = df.groupby("nm_id")["price"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )

    df["price_diff_1"] = df["price"] - df["lag_price_1"]
    df["price_ratio_7"] = df["price"] / (df["rolling_mean_price_7"] + 1e-6)
    df["price_change_pct"] = df["price"] / df.groupby("nm_id")[
        "price"
    ].transform("mean")

    return df


def add_global_lags(df: pd.DataFrame) -> pd.DataFrame:
    daily_sales = df.groupby("dt")["qty"].sum().reset_index()
    daily_sales.columns = ["dt", "global_sales"]

    df = df.merge(daily_sales, on="dt", how="left")

    for lag in [1, 7, 14]:
        df[f"global_lag_{lag}"] = df["global_sales"].shift(lag)

    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    promo_stats = (
        df.groupby("nm_id")
        .apply(
            lambda x: pd.Series(
                {
                    "mean_qty_promo": x[x["is_promo"] == 1]["qty"].mean(),
                    "mean_qty_no_promo": x[x["is_promo"] == 0]["qty"].mean(),
                }
            )
        )
        .fillna(0)
    )

    promo_stats["promo_uplift_ratio"] = promo_stats["mean_qty_promo"] / (
        promo_stats["mean_qty_no_promo"] + 1e-6
    )
    promo_stats["promo_uplift"] = np.log1p(
        promo_stats["mean_qty_promo"]
    ) - np.log1p(promo_stats["mean_qty_no_promo"])
    promo_stats["promo_uplift_diff"] = (
        promo_stats["mean_qty_promo"] - promo_stats["mean_qty_no_promo"]
    )

    df = df.merge(promo_stats, on="nm_id", how="left")

    return df


def add_product_clusters(df: pd.DataFrame, n_clusters: int = 3):
    cluster_stats = (
        df.groupby("nm_id")
        .agg(
            {
                "item_mean_qty": "first",
                "item_std_qty": "first",
                "item_nonzero_ratio": "first",
            }
        )
        .fillna(0)
    )

    cluster_stats.columns = ["mean_qty", "volatility", "sale_freq"]
    cluster_stats["zero_ratio"] = 1 - cluster_stats["sale_freq"]

    cluster_stats["cluster_id"] = KMeans(
        n_clusters=n_clusters, random_state=42, n_init="auto"
    ).fit_predict(cluster_stats[["mean_qty", "volatility", "zero_ratio"]])

    return df.merge(cluster_stats[["cluster_id"]], on="nm_id", how="left")


def add_product_clusters_v2(df: pd.DataFrame, n_clusters: int = 4):
    cluster_stats = (
        df.groupby("nm_id")["qty"]
        .agg(
            log_mean_qty=lambda x: np.log1p(x.mean()),
            volatility=lambda x: x.std(),
            zero_ratio=lambda x: (x == 0).mean(),
        )
        .fillna(0)
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_stats)

    cluster_stats["cluster_id"] = KMeans(
        n_clusters=n_clusters, random_state=42, n_init="auto"
    ).fit_predict(scaled)

    return df.merge(cluster_stats[["cluster_id"]], on="nm_id", how="left")


def add_product_clusters_v3(df: pd.DataFrame, n_clusters: int = 4):
    train_only = df[df["__is_train"] == 1].copy()

    cluster_stats = (
        train_only.groupby("nm_id")
        .agg(
            log_mean_qty=("qty", lambda x: np.log1p(x.mean())),
            zero_ratio=("qty", lambda x: (x == 0).mean()),
            cv=("qty", lambda x: x.std() / (x.mean() + 1e-6)),
            promo_ratio=("is_promo", "mean"),
            trend_slope=(
                "qty",
                lambda x: np.polyfit(np.arange(len(x)), np.log1p(x), 1)[0]
                if len(x) > 1
                else 0,
            ),
            nonzero_run_mean=(
                "qty",
                lambda x: (
                    np.mean(
                        [len(list(g)) for v, g in groupby((x.values > 0)) if v]
                    )
                    if np.any(x.values > 0)
                    else 0
                ),
            ),
            obs_count=("qty", "count"),
        )
        .fillna(0)
    )

    features = [
        "log_mean_qty",
        "zero_ratio",
        "cv",
        "promo_ratio",
        "trend_slope",
        "nonzero_run_mean",
    ]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_stats[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)

    cluster_stats["cluster_id"] = kmeans.fit_predict(scaled)

    cluster_stats["cluster_id"] = cluster_stats["cluster_id"].astype("int")

    df = df.merge(cluster_stats[["cluster_id"]], on="nm_id", how="left")

    df["cluster_id"] = df["cluster_id"].fillna(-1).astype("int")

    return df


def add_product_clusters_v4(df: pd.DataFrame, n_clusters: int = 4):
    train_only = df[df["__is_train"] == 1].copy()

    cluster_stats = (
        train_only.groupby("nm_id")
        .agg(
            log_mean_qty=("qty", lambda x: np.log1p(x.mean())),
            zero_ratio=("qty", lambda x: (x == 0).mean()),
            cv=("qty", lambda x: x.std() / (x.mean() + 1e-6)),
            promo_ratio=("is_promo", "mean"),
            trend_slope=(
                "qty",
                lambda x: np.polyfit(np.arange(len(x)), np.log1p(x), 1)[0]
                if len(x) > 1
                else 0,
            ),
            nonzero_run_mean=(
                "qty",
                lambda x: (
                    np.mean(
                        [len(list(g)) for v, g in groupby(x.values > 0) if v]
                    )
                    if np.any(x.values > 0)
                    else 0
                ),
            ),
            obs_count=("qty", "count"),
        )
        .fillna(0)
    )

    cluster_stats["log_obs_count"] = np.log1p(cluster_stats["obs_count"])

    features = [
        "log_mean_qty",
        "zero_ratio",
        "cv",
        "promo_ratio",
        "trend_slope",
        "nonzero_run_mean",
        "log_obs_count",
    ]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_stats[features])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
    )

    cluster_stats["cluster_id"] = kmeans.fit_predict(scaled).astype(int)

    df = df.merge(
        cluster_stats[["cluster_id"]],
        on="nm_id",
        how="left",
    )

    df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)

    return df


def add_product_stats(df: pd.DataFrame) -> pd.DataFrame:
    train_only = df[df["__is_train"] == 1].copy()

    product_stats = (
        train_only.groupby("nm_id")["qty"]
        .agg(
            log_mean_qty=lambda x: np.log1p(x.mean()),
            cv=lambda x: x.std() / (x.mean() + 1e-6) if x.mean() > 0 else 0,
            obs_count=len,
        )
        .fillna(0)
    )

    product_stats["promo_ratio"] = train_only.groupby("nm_id")[
        "is_promo"
    ].mean()

    return df.merge(product_stats, on="nm_id", how="left")


def handle_promo_uplift_outliers(
    df: pd.DataFrame,
    col: str = "promo_uplift_ratio",
    low_perc: float = 0.01,
    high_perc: float = 0.99,
) -> pd.DataFrame:
    """Обрабатывает выбросы в promo_uplift_ratio.

    Параметры:
    - df: DataFrame
    - col: имя столбца с выбросами
    - low_perc, high_perc: процентили для обрезки (по умолчанию 1% и 99%)

    Returns:
    DataFrame с обработанным столбцом
    """
    low, high = np.percentile(df[col], [low_perc, high_perc])
    df[col] = np.clip(df[col], low, high)
    return df


def scale_features(
    df: pd.DataFrame, cols: list = ["price", "prev_leftovers", "global_sales"]
) -> pd.DataFrame:
    """Нормализует указанные признаки с помощью RobustScaler."""
    scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df
