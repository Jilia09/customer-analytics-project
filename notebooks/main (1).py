import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

conn = sqlite3.connect("online_store.db")
query = """
SELECT 
    -- 1. Таблиця customers
    c.customer_id,
    c.signup_date,
    c.gender,
    c.city AS customer_city,
    c.country AS customer_country,
    c.segment,

    -- 2. Таблиця customer_pii (персональні дані)
    pii.phone,
    pii.full_address,
    pii.national_id,
    pii.card_last4,

    -- 3. Таблиця orders
    o.order_id,
    o.order_date,
    o.status AS order_status,

    -- 4. Таблиця payments (сума платежу)
    p.amount AS payment_amount,

    -- 5. Таблиця returns (суми повернень)
    r.refund_amount,

    -- 6. Таблиця reviews (оцінки клієнтів)
    rev.rating AS review_rating

FROM customers c
LEFT JOIN customer_pii pii ON c.customer_id = pii.customer_id
LEFT JOIN orders o         ON c.customer_id = o.customer_id
LEFT JOIN payments p       ON o.order_id = p.order_id
LEFT JOIN returns r        ON o.order_id = r.order_id
-- Зв'язок за customer_id згідно з вашою структурою БД
LEFT JOIN reviews rev      ON c.customer_id = rev.customer_id 
"""

df_raw = pd.read_sql_query(query, conn)
conn.close()

df_raw["signup_date"] = pd.to_datetime(df_raw["signup_date"])
df_raw["order_date"] = pd.to_datetime(df_raw["order_date"])
df_raw["payment_amount"] = df_raw["payment_amount"].fillna(0)
df_raw["refund_amount"] = df_raw["refund_amount"].fillna(0)
df_raw["net_amount"] = df_raw["payment_amount"] - df_raw["refund_amount"]

snapshot_date = df_raw["order_date"].max() + pd.Timedelta(days=1)

df_rfm = (
    df_raw.groupby("customer_id")
    .agg(
        {
            "order_date": lambda x: (snapshot_date - x.max()).days if pd.notna(x.max()) else np.nan,
            "order_id": lambda x: x.nunique() if pd.notna(x.max()) else 0,
            "net_amount": "sum",
        }
    )
    .reset_index()
)
df_rfm.columns = ["customer_id", "Recency", "Frequency", "Monetary"]
df_rfm["Monetary"] = df_rfm["Monetary"].round(2)
max_recency = df_rfm["Recency"].max()
df_rfm["Recency"] = df_rfm["Recency"].fillna(max_recency + 30 if pd.notna(max_recency) else 365)

df_rfm.to_csv("rfm_result.csv", index=False)
df_cohort = df_raw.dropna(subset=["order_date"]).copy()
df_cohort = df_cohort[df_cohort["order_date"] >= df_cohort["signup_date"]]
df_cohort["cohort_month"] = df_cohort["signup_date"].dt.to_period("M")
df_cohort["order_month"] = df_cohort["order_date"].dt.to_period("M")
df_cohort["cohort_index"] = (
        (df_cohort["order_month"].dt.year - df_cohort["cohort_month"].dt.year) * 12
        + (df_cohort["order_month"].dt.month - df_cohort["cohort_month"].dt.month)
)
cohort_data = (
    df_cohort.groupby(["cohort_month", "cohort_index"])["customer_id"]
    .nunique()
    .reset_index()
)
cohort_pivot = cohort_data.pivot(
    index="cohort_month", columns="cohort_index", values="customer_id"
)
cohort_sizes = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0).round(4) * 100
max_months = 12
retention_truncated = retention_matrix.iloc[:, :max_months]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Розподіл метрик RFM за клієнтами", fontsize=16, fontweight="bold")

sns.histplot(df_rfm["Recency"], kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Recency (Днів від останнього замовлення)")
axes[0].set_xlabel("Дні")
axes[0].set_ylabel("Кількість клієнтів")

sns.histplot(df_rfm["Frequency"], kde=False, ax=axes[1], color="salmon", bins=min(15, df_rfm["Frequency"].nunique()))
axes[1].set_title("Frequency (Кількість замовлень)")
axes[1].set_xlabel("Замовлення")
axes[1].set_ylabel("Кількість клієнтів")

sns.histplot(df_rfm["Monetary"], kde=True, ax=axes[2], color="lightgreen")
axes[2].set_title("Monetary (Загальна сума покупок)")
axes[2].set_xlabel("Сума")
axes[2].set_ylabel("Кількість клієнтів")

plt.tight_layout()
plt.savefig("rfm_distributions.png", dpi=300)
plt.show()

plt.figure(figsize=(16, 12))
plt.title(
    f"Когортний аналіз: Retention Rate за перші {max_months} місяців (в %)\nКогорти сформовано за місяцем реєстрації (signup_date)",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
sns.heatmap(
    retention_truncated,
    annot=True,
    fmt=".1f",
    annot_kws={"size": 8, "weight": "bold"},
    cmap="YlGnBu",
    linewidths=0.5,
    vmax=25,
    cbar_kws={"label": "Частка клієнтів, що повернулися (%)", "shrink": 0.8},
)
plt.xlabel("Місяці після реєстрації (Cohort Index)", fontsize=12, labelpad=10)
plt.ylabel("Когорта (Місяць реєстрації signup_date)", fontsize=12, labelpad=10)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("cohort_retention_beautiful.png", dpi=300)
plt.show()