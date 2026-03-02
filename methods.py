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

    return df


def add_price_features(full_df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет динамику цены.
    Работает на объединённом full_df (train+val+test).
    Не создаёт leakage, так как использует shift.
    """

    df = full_df.copy()
    df = df.sort_values(["nm_id", "dt"])

    # лаг цены
    df["lag_price_1"] = df.groupby("nm_id")["price"].shift(1)

    # скользящее среднее цены (без текущего дня)
    df["rolling_mean_price_7"] = df.groupby("nm_id")["price"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )

    # разница
    df["price_diff_1"] = df["price"] - df["lag_price_1"]

    # отношение к последней неделе
    df["price_ratio_7"] = df["price"] / (df["rolling_mean_price_7"] + 1e-6)

    # отклонение от среднего по товару (по всей истории ДО текущего дня)
    df["expanding_mean_price"] = df.groupby("nm_id")["price"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df["price_change_pct"] = df["price"] / (df["expanding_mean_price"] + 1e-6)

    return df


def add_global_lags(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет глобальные признаки продаж:
    - global_sales
    - global_lag_1, global_lag_7, global_lag_14
    
    full_df должен содержать служебные флаги:
    __is_train, __is_val
    
    Признаки безопасны — нет утечки будущей информации.
    """

    df = full_df.copy().sort_values("dt")

    # -------------------------------
    # 1️⃣ Глобальные продажи по дням для train
    # -------------------------------
    train_mask = df["__is_train"] == 1
    global_sales_train = (
        df.loc[train_mask]
        .groupby("dt")["qty"]
        .sum()
        .rename("global_sales")
    )

    # -------------------------------
    # 2️⃣ Создаём колонку global_sales и заполняем для всех дней
    # -------------------------------
    df["global_sales"] = df["dt"].map(global_sales_train).fillna(0)

    # -------------------------------
    # 3️⃣ Добавляем лаги
    # -------------------------------
    for lag in [1, 7, 14]:
        df[f"global_lag_{lag}"] = df["global_sales"].shift(lag).fillna(0)

    return df



def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    train_only = df[(df["__is_train"] == 1) & (df["__is_val"] == 0)].copy()
    promo_stats = (
        train_only.groupby("nm_id")
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
    df[["promo_uplift_ratio", "promo_uplift", "promo_uplift_diff"]] = df[
        ["promo_uplift_ratio", "promo_uplift", "promo_uplift_diff"]
    ].fillna(0)

    return df


def add_seasonality_flags(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет high_season / cold_season.
    Статистики считаются ТОЛЬКО по train-части (где qty notna() и __is_val == 0).
    """

    df = full_df.copy()

    # 1. Берём только train (без val и test)
    train_part = df[(df["qty"].notna()) & (df["__is_val"] == 0)].copy()

    # 2. Считаем среднюю дневную продажу по месяцам
    monthly_stats = (
        train_part.groupby(["nm_id", "month"])["qty"].mean().reset_index()
    )

    # 3. Среднее и std по месяцам внутри каждого товара
    monthly_stats["mean_nm"] = monthly_stats.groupby("nm_id")["qty"].transform(
        "mean"
    )
    monthly_stats["std_nm"] = monthly_stats.groupby("nm_id")["qty"].transform(
        "std"
    )

    # чтобы избежать деления на 0
    monthly_stats["std_nm"] = monthly_stats["std_nm"].replace(0, 1e-6)

    # 4. Z-score
    monthly_stats["z"] = (
        monthly_stats["qty"] - monthly_stats["mean_nm"]
    ) / monthly_stats["std_nm"]

    monthly_stats["high_season"] = (monthly_stats["z"] > 1).astype("int8")
    monthly_stats["cold_season"] = (monthly_stats["z"] < -1).astype("int8")

    # 5. Merge обратно во весь full_df
    df = df.merge(
        monthly_stats[["nm_id", "month", "high_season", "cold_season"]],
        on=["nm_id", "month"],
        how="left",
    )

    df["high_season"] = df["high_season"].fillna(0).astype("int8")
    df["cold_season"] = df["cold_season"].fillna(0).astype("int8")

    return df


def add_product_activity_features(full_df: pd.DataFrame) -> pd.DataFrame:
    df = full_df.copy().sort_values(["nm_id", "dt"])

    # заменяем NaN qty на 0 только для расчётов
    qty_temp = df["qty"].fillna(0)

    # cumulative факт продаж до текущего дня
    df["was_sale"] = qty_temp.groupby(df["nm_id"]).transform(
        lambda x: x.shift(1).cumsum()
    )

    df["was_sale"] = (df["was_sale"] > 0).astype("int8")

    # продажи за 30 дней
    df["sales_last_30"] = qty_temp.groupby(df["nm_id"]).transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).sum()
    )

    # доля дней с продажами
    df["sale_days_last_30"] = qty_temp.groupby(df["nm_id"]).transform(
        lambda x: (x.shift(1) > 0).rolling(30, min_periods=1).mean()
    )

    def compute_days_since_last_sale(x):
        x_shifted = x.shift(1).fillna(0)
        result = []
        counter = 0

        for val in x_shifted:
            if val > 0:
                counter = 0
            else:
                counter += 1
            result.append(counter)

        return pd.Series(result, index=x.index)

    df["days_since_last_sale"] = (
        qty_temp.groupby(df["nm_id"])
        .apply(compute_days_since_last_sale)
        .reset_index(level=0, drop=True)
    )

    return df


def add_leftover_change_features(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет:
    - leftovers_changed_lag
    - days_since_leftover_change
    
    Использует только прошлые данные (через shift).
    Без leakage.
    """

    df = full_df.copy().sort_values(["nm_id", "dt"])

    # 1️⃣ Было ли изменение вчера
    df["leftovers_changed"] = (
        df.groupby("nm_id")["prev_leftovers"]
        .diff()
        .ne(0)
        .astype("int8")
    )

    # 2️⃣ Используем ТОЛЬКО прошлую информацию
    df["leftovers_changed_lag"] = (
        df.groupby("nm_id")["leftovers_changed"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    # 3️⃣ Считаем дни с прошлого изменения (на основе lag!)
    def compute_days_since_change(x):
        result = []
        counter = 0
        for val in x:
            if val == 1:
                counter = 0
            else:
                counter += 1
            result.append(counter)
        return pd.Series(result, index=x.index)

    df["days_since_leftover_change"] = (
        df.groupby("nm_id")["leftovers_changed_lag"]
        .apply(compute_days_since_change)
        .reset_index(level=0, drop=True)
    )

    df["days_since_leftover_change"] = (
        df["days_since_leftover_change"].fillna(0)
    )

    return df

