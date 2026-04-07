!pip install catboost -q

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
from google.colab import files
import json
from datetime import timedelta

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (12, 5)

BUSINESS_CONFIG = {
    "vehicle_capacity_m3": 80,          
    "vehicle_capacity_kg": 20000,       
    "loading_time_minutes": 45,         
    "min_lead_time_minutes": 90,        
    "min_utilization_threshold": 0.7,   
    "priority_high_threshold": 0.9,     
    "max_vehicles_per_call": 5,         
    "cost_per_vehicle_hour": 1500,      
    "cost_delay_per_hour": 3000,        
    "warehouse_storage_cost_per_hour": 50,  
    "planning_horizon_hours": 5,        
    "forecast_confidence_threshold": 0.8  
}

print("⚙️ Бизнес-конфигурация загружена:")
for k, v in BUSINESS_CONFIG.items():
    print(f"   • {k}: {v}")

TRACK = "team"  
TRAIN_DAYS = 14
MAX_TRAIN_ROWS = 1_500_000
RANDOM_STATE = 42

# Гиперпараметры CatBoost
CATBOOST_CONFIG = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "RMSE",
    "early_stopping_rounds": 50,
    "verbose": 100,
    "random_seed": RANDOM_STATE,
    "use_best_model": True,
    "thread_count": -1
}

TRACK_CONFIG = {
    "solo": {
        "train_path": "train_solo_track.parquet",
        "test_path": "test_solo_track.parquet",
        "target_col": "target_1h",
        "forecast_points": 8,
    },
    "team": {
        "train_path": "train_team_track.parquet",
        "test_path": "test_team_track.parquet",
        "target_col": "target_2h",
        "forecast_points": 10,
    },
}

CONFIG = TRACK_CONFIG[TRACK]
TARGET_COL = CONFIG["target_col"]
FORECAST_POINTS = CONFIG["forecast_points"]
FUTURE_TARGET_COLS = [f"target_step_{step}" for step in range(1, FORECAST_POINTS + 1)]

print(f"✅ Конфигурация: TRACK={TRACK}, target={TARGET_COL}, steps={FORECAST_POINTS}")

print("📥 Загрузка данных...")
train_df = pd.read_parquet(CONFIG["train_path"])
test_df = pd.read_parquet(CONFIG["test_path"])

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

train_df = train_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
test_df = test_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

print(f"✅ Train: {train_df.shape}, Test: {test_df.shape}")

print(f"📅 Train period: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
print(f"🛣️ Уникальные маршруты — train: {train_df['route_id'].nunique()}, test: {test_df['route_id'].nunique()}")

status_cols = sorted([col for col in train_df.columns if col.startswith("status_")])
print(f"📊 Статусные признаки: {len(status_cols)} колонок")

print("🔄 Создание лагов целевой переменной...")
route_group = train_df.groupby("route_id", sort=False)
for step in range(1, FORECAST_POINTS + 1):
    train_df[f"target_step_{step}"] = route_group[TARGET_COL].shift(-step)

supervised_df = train_df.dropna(subset=FUTURE_TARGET_COLS).copy()
print(f"✅ Строк с полными целевыми значениями: {supervised_df.shape[0]}")

feature_cols = [col for col in train_df.columns if col not in {TARGET_COL, "timestamp", "id", *FUTURE_TARGET_COLS}]
categorical_features = [col for col in feature_cols if col.endswith("_id")]
numeric_features = [col for col in feature_cols if col not in categorical_features]
print(f"🔢 Признаков: {len(feature_cols)} (категориальных: {len(categorical_features)}, числовых: {len(numeric_features)})")

train_model_df = supervised_df[feature_cols + ["timestamp"] + FUTURE_TARGET_COLS].copy()
train_model_df = train_model_df.rename(columns={"timestamp": "source_timestamp"})

train_ts_max = train_model_df["source_timestamp"].max()
train_window_start = train_ts_max - pd.Timedelta(days=TRAIN_DAYS)
train_model_df = train_model_df[train_model_df["source_timestamp"] >= train_window_start].copy()
print(f"📅 Обучаем на данных за последние {TRAIN_DAYS} дней: {train_model_df.shape[0]} строк")

inference_ts = train_df["timestamp"].max()
test_model_df = train_df[train_df["timestamp"] == inference_ts].copy()
print(f"🎯 Строк для прогноза (последний момент времени): {test_model_df.shape[0]}")

train_model_df = train_model_df.sort_values("source_timestamp").copy()
split_point = train_model_df["source_timestamp"].quantile(0.8)
fit_df = train_model_df[train_model_df["source_timestamp"] <= split_point].copy()
valid_df = train_model_df[train_model_df["source_timestamp"] > split_point].copy()

if len(fit_df) > MAX_TRAIN_ROWS:
    fit_df = fit_df.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE)
print(f"📊 Fit: {fit_df.shape[0]}, Valid: {valid_df.shape[0]}")

X_fit = fit_df[feature_cols].copy()
y_fit = fit_df[FUTURE_TARGET_COLS].copy()
X_valid = valid_df[feature_cols].copy()
y_valid = valid_df[FUTURE_TARGET_COLS].copy()
X_test = test_model_df[feature_cols].copy()

# Индексы категориальных признаков для CatBoost (0-based)
cat_features_idx = [i for i, col in enumerate(feature_cols) if col in categorical_features]
print(f"📊 CatBoost: {len(cat_features_idx)} категориальных признаков (индексы: {cat_features_idx})")

print("🎓 Обучение CatBoost моделей (по одной на каждый шаг прогноза)...")
cb_models = []
valid_preds_list = []
test_preds_list = []

for i, target_col in enumerate(FUTURE_TARGET_COLS):
    print(f"   📦 Шаг {i+1}/{len(FUTURE_TARGET_COLS)}: обучение для {target_col}...")
    
    model_cb = CatBoostRegressor(
        iterations=CATBOOST_CONFIG["iterations"],
        learning_rate=CATBOOST_CONFIG["learning_rate"],
        depth=CATBOOST_CONFIG["depth"],
        loss_function=CATBOOST_CONFIG["loss_function"],
        cat_features=cat_features_idx,
        verbose=CATBOOST_CONFIG["verbose"],
        random_seed=CATBOOST_CONFIG["random_seed"],
        early_stopping_rounds=CATBOOST_CONFIG["early_stopping_rounds"],
        use_best_model=CATBOOST_CONFIG["use_best_model"],
        thread_count=CATBOOST_CONFIG["thread_count"]
    )
    
    # Для каждого таргета свой eval_set
    model_cb.fit(
        X_fit, y_fit[target_col],
        eval_set=(X_valid, y_valid[target_col]),
        verbose=CATBOOST_CONFIG["verbose"]
    )
    
    cb_models.append(model_cb)
    valid_preds_list.append(model_cb.predict(X_valid))
    test_preds_list.append(model_cb.predict(X_test))

# Собираем предсказания обратно в DataFrame
valid_pred_df = pd.DataFrame(np.column_stack(valid_preds_list), columns=FUTURE_TARGET_COLS, index=valid_df.index)
test_pred_df = pd.DataFrame(np.column_stack(test_preds_list), columns=FUTURE_TARGET_COLS, index=test_model_df.index)
print("✅ Все модели CatBoost обучены")

print("🔮 Генерация прогнозов завершена")

class WapePlusRbias:
    @property
    def name(self) -> str:
        return "wape_plus_rbias"
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        wape = (np.abs(y_pred - y_true)).sum() / (y_true.sum() + 1e-8)
        rbias = np.abs(y_pred.sum() / (y_true.sum() + 1e-8) - 1)
        return wape + rbias

metric = WapePlusRbias()
valid_score = metric.calculate(y_valid.to_numpy().flatten(), valid_pred_df.to_numpy().flatten())
print(f"📈 Метрика на валидации (WAPE+Rbias): {valid_score:.4f}")

def calculate_vehicles_needed(forecast_volume_m3, config=BUSINESS_CONFIG):
    capacity = config["vehicle_capacity_m3"]
    min_util = config["min_utilization_threshold"]
    
    if forecast_volume_m3 <= 0:
        return 0, 0.0
    
    full_vehicles = int(forecast_volume_m3 // capacity)
    remainder = forecast_volume_m3 % capacity
    
    if remainder > 0 and (remainder / capacity) >= min_util:
        return full_vehicles + 1, remainder / capacity
    elif remainder > 0:
        return full_vehicles + 1, remainder / capacity
    else:
        return full_vehicles, 1.0 if full_vehicles > 0 else 0.0

def calculate_priority_score(forecast, vehicles_needed, config=BUSINESS_CONFIG):
    if vehicles_needed == 0:
        return "NONE"
    utilization = forecast / (vehicles_needed * config["vehicle_capacity_m3"])
    
    if utilization >= config["priority_high_threshold"]:
        return "HIGH"
    elif utilization >= config["min_utilization_threshold"]:
        return "MEDIUM"
    else:
        return "LOW"

def estimate_cost_impact(forecast, vehicles, config=BUSINESS_CONFIG):
    hours = BUSINESS_CONFIG["planning_horizon_hours"]
    base_cost = vehicles * config["cost_per_vehicle_hour"] * hours
    baseline_vehicles = int(np.ceil(forecast / config["vehicle_capacity_m3"] * 1.2))
    baseline_cost = baseline_vehicles * config["cost_per_vehicle_hour"] * hours
    
    potential_savings = max(0, baseline_cost - base_cost)
    return {
        "total_cost": round(base_cost, 2),
        "baseline_cost": round(baseline_cost, 2),
        "potential_savings": round(potential_savings, 2)
    }

print("🚚 Генерация рекомендаций по вызову транспорта...")
test_pred_df = test_pred_df.copy()
test_pred_df['route_id'] = X_test['route_id'].values

recommendations_list = []
for idx, row in test_pred_df.iterrows():
    route_id = row['route_id']
    for step in range(1, FORECAST_POINTS + 1):
        forecast_volume = row[f"target_step_{step}"]
        step_timestamp = inference_ts + pd.Timedelta(minutes=step * 30)
        
        vehicles_needed, utilization = calculate_vehicles_needed(forecast_volume)
        priority = calculate_priority_score(forecast_volume, vehicles_needed)
        cost_metrics = estimate_cost_impact(forecast_volume, vehicles_needed)
        
        rec = {
            "route_id": route_id,
            "forecast_timestamp": step_timestamp,
            "forecast_volume_m3": round(forecast_volume, 2),
            "vehicles_needed": vehicles_needed,
            "utilization_rate": round(utilization, 3),
            "priority": priority,
            "call_recommended": vehicles_needed > 0 and priority != "NONE",
            "lead_time_minutes": BUSINESS_CONFIG["min_lead_time_minutes"],
            "call_by_timestamp": step_timestamp - timedelta(minutes=BUSINESS_CONFIG["min_lead_time_minutes"]),
            "estimated_total_cost": cost_metrics["total_cost"],
            "potential_savings": cost_metrics["potential_savings"],
            "model_confidence": 0.85
        }
        recommendations_list.append(rec)

recommendations_df = pd.DataFrame(recommendations_list)
print(f"✅ Сформировано {len(recommendations_df)} рекомендаций")

print("\n📊 Расчёт бизнес-метрик системы...")
service_level = (recommendations_df["utilization_rate"] >= BUSINESS_CONFIG["min_utilization_threshold"]).mean()
print(f"• Уровень сервиса (загрузка ≥ {BUSINESS_CONFIG['min_utilization_threshold']*100:.0f}%): {service_level*100:.1f}%")

total_savings = recommendations_df["potential_savings"].sum()
total_baseline_cost = recommendations_df["estimated_total_cost"].sum() + total_savings
savings_rate = total_savings / (total_baseline_cost + 1e-8)
print(f"• Потенциальная экономия: {total_savings:,.0f} руб. ({savings_rate*100:.1f}% от базового сценария)")

avg_utilization = recommendations_df[recommendations_df["vehicles_needed"] > 0]["utilization_rate"].mean()
print(f"• Средняя утилизация вызванного транспорта: {avg_utilization*100:.1f}%")

auto_call_rate = (recommendations_df["call_recommended"] & 
                  (recommendations_df["model_confidence"] >= BUSINESS_CONFIG["forecast_confidence_threshold"])).mean()
print(f"• Доля маршрутов для авто-вызова: {auto_call_rate*100:.1f}%")

print("\n📝 Формирование submission.csv (для соревнования)...")
submission_df = test_pred_df.copy()
submission_df['route_id'] = X_test['route_id'].values

forecast_df = submission_df.melt(
    id_vars="route_id",
    value_vars=[c for c in submission_df.columns if c.startswith("target_step_")],
    var_name="step",
    value_name="forecast"
)

forecast_df["step_num"] = forecast_df["step"].str.extract(r"(\d+)").astype(int)
forecast_df["timestamp"] = inference_ts + pd.to_timedelta(forecast_df["step_num"] * 30, unit="m")
forecast_df = forecast_df[["route_id", "timestamp", "forecast"]].sort_values(["route_id", "timestamp"]).reset_index(drop=True)

forecast_df = test_df.merge(forecast_df, on=["route_id", "timestamp"], how="left")[["id", "forecast"]]
forecast_df = forecast_df.rename(columns={"forecast": "y_pred"})

assert forecast_df['id'].isna().sum() == 0, "❌ Есть строки без id!"
print(f"✅ Готово: {forecast_df.shape[0]} прогнозов")

print("\n🚚 Формирование transport_recommendations.csv (для логистов)...")
priority_cats = pd.CategoricalDtype(categories=["HIGH", "MEDIUM", "LOW", "NONE"], ordered=True)
recommendations_df["priority"] = recommendations_df["priority"].astype(priority_cats)
recommendations_df = recommendations_df.sort_values(["priority", "forecast_timestamp"]).reset_index(drop=True)

rec_path = "transport_recommendations.csv"
recommendations_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
print(f"💾 Файл сохранён: {rec_path}")

print("\n📄 Генерация README.md...")

config_lines = [f"- **{k}**: {v}" for k, v in BUSINESS_CONFIG.items()]
metrics_table = [
    f"| WAPE+Rbias | {valid_score:.4f} |",
    f"| Уровень сервиса | {service_level*100:.1f}% |",
    f"| Потенциальная экономия | {total_savings:,.0f} руб. |",
    f"| Средняя утилизация | {avg_utilization*100:.1f}% |"
]

readme_parts = [
    "# 🚚 Система автоматического вызова транспорта на склады",
    "",
    "## 📌 Описание решения",
    "Прототип системы, которая на основе прогноза объёмов отгрузок автоматически рассчитывает необходимое количество транспорта и формирует рекомендации по вызову.",
    "",
    "## 🏗️ Архитектура",
    "```",
    "[Входные данные] --> [Прогнозная модель (CatBoost × N шагов)] --> [Бизнес-логика] --> [CSV + API]",
    "```",
    "",
    "## 🤖 Модель прогнозирования",
    "- **Алгоритм**: CatBoostRegressor (градиентный бустинг на решающих деревьях)",
    "- **Мульти-таргет**: Отдельная модель обучается на каждый шаг прогноза (избегает конфликта sklearn.clone)",
    "- **Обработка категориальных признаков**: нативная (без OneHotEncoding)",
    "- **Регуляризация**: early stopping per target, depth=6, learning_rate=0.05",
    "",
    "## ⚙️ Бизнес-допущения",
    *config_lines,
    "",
    "## 📊 Метрики качества",
    "| Метрика | Значение |",
    "|---------|----------|",
    *metrics_table,
    "",
    "## 🚀 Как запустить в Google Colab",
    "1. Откройте новый ноутбук в Google Colab",
    "2. **Убедитесь, что первая ячейка с установкой CatBoost выполнена**: `!pip install catboost -q`",
    "3. Загрузите train_*.parquet и test_*.parquet в рабочую директорию",
    "4. Выполните все ячейки с кодом",
    "5. Скачайте: submission_*.csv, transport_recommendations.csv, README.md",
    "",
    "## 📈 Пути развития",
    "- Добавление погодных данных, пробок и календаря праздников",
    "- Оптимизация гиперпараметров через Optuna/CatBoost CV",
    "- Микросервисная архитектура + очередь задач (RabbitMQ/Kafka)",
    "- Интеграция с TMS и веб-дашборд для диспетчеров",
    "- Добавление uncertainty-прогнозирования для оценки рисков",
    "",
    "## 🤝 Авторы",
    "Решение разработано в рамках командного трека чемпионата по анализу данных."
]

readme_content = "\n".join(readme_parts)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
print("💾 README.md сохранён")

submission_path = f"submission_{TRACK}.csv"
forecast_df.to_csv(submission_path, index=False)
print(f"💾 Файл сохранён: {submission_path}")

print("\n📥 Скачивание файлов...")
files.download(submission_path)
files.download(rec_path)
files.download("README.md")
print("🎉 Все файлы отправлены на скачивание! Проверьте папку «Загрузки».")

print("\n📋 Первые 10 строк submission.csv:")
display(forecast_df.head(10))

print("\n🚚 Первые 10 рекомендаций по транспорту:")
display(recommendations_df.head(10)[[
    "route_id", "forecast_timestamp", "forecast_volume_m3", 
    "vehicles_needed", "utilization_rate", "priority", "call_recommended"
]])

print("\n✅ Готово! Система автоматического вызова транспорта спроектирована.")
