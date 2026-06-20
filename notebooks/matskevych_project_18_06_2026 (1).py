import os
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, 'online_store.db')


def get_connection():
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"Database not found {DB_PATH} ")
    return sqlite3.connect(DB_PATH)

def analyze_returns_and_refunds(conn):
    print("\n" + "=" * 50)
    print("1. АНАЛІЗ ПОВЕРНЕНЬ ТА ВІДШКОДУВАНЬ")
    print("=" * 50)

    query_returns = """
    SELECT 
        c.name AS category_name,
        r.reason AS return_reason,
        oi.quantity * oi.unit_price AS refund_amount
    FROM returns r
    JOIN order_items oi ON r.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN categories c ON p.category_id = c.category_id
    """
    df_ret = pd.read_sql_query(query_returns, conn)

    total_refunds = df_ret['refund_amount'].sum()
    print(f"Загальна сума відшкодувань (Total Refunds): {total_refunds:,.2f}")

    print("\nТоп-5 категорій за часткою від усіх повернень:")
    cat_shares = df_ret['category_name'].value_counts(normalize=True) * 100
    print(cat_shares.head(5).map('{:.2f}%'.format).to_string())

    print("\nРозподіл причин повернень (Return Reasons):")
    reasons = df_ret['return_reason'].value_counts(normalize=True) * 100
    print(reasons.map('{:.2f}%'.format).to_string())

    return df_ret


def analyze_reviews_and_loyalty(conn):
    print("\n" + "=" * 50)
    print("2. ВІДГУКИ ТА ЛОЯЛЬНІСТЬ КЛІЄНТІВ")
    print("=" * 50)

    query_reviews = """
    SELECT 
        r.rating,
        r.customer_id,
        c.name AS category_name
    FROM reviews r
    JOIN products p ON r.product_id = p.product_id
    JOIN categories c ON p.category_id = c.category_id
    """
    df_rev = pd.read_sql_query(query_reviews, conn)

    print("Розподіл оцінок у відгуках (Rating Distribution):")
    rating_dist = df_rev['rating'].value_counts(normalize=True).sort_index() * 100
    print(rating_dist.map('{:.2f}%'.format).to_string())

    print("\nСередній рейтинг у розрізі топ-5 категорій:")
    avg_rating = df_rev.groupby('category_name')['rating'].mean().sort_values(ascending=False)
    print(avg_rating.head(5).round(2))

    query_loyalty = """
    SELECT 
        rev.customer_id,
        AVG(rev.rating) as avg_customer_rating,
        COUNT(DISTINCT o.order_id) as total_orders
    FROM reviews rev
    JOIN orders o ON rev.customer_id = o.customer_id
    GROUP BY rev.customer_id
    """
    df_loyalty = pd.read_sql_query(query_loyalty, conn)

    print("\nЗв'язок між середньою оцінкою клієнта та кількістю його замовлень:")

    df_loyalty['rating_group'] = df_loyalty['avg_customer_rating'].round()
    loyalty_summary = df_loyalty.groupby('rating_group')['total_orders'].mean()
    print(loyalty_summary.round(2).to_string())

    return df_rev, df_loyalty


def generate_visualizations(df_ret, df_rev):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax0 = sns.countplot(data=df_rev, x='rating', ax=axes[0], palette="viridis")
    axes[0].set_title("Розподіл оцінок користувачів (Reviews)", fontsize=14)
    axes[0].set_xlabel("Оцінка (Rating)", fontsize=12)
    axes[0].set_ylabel("Кількість відгуків", fontsize=12)

    for container in ax0.containers:
        ax0.bar_label(container, fmt='%d', label_type='edge', padding=4, fontweight='bold', color='#333333')

    if not df_ret.empty:
        order_reasons = df_ret['return_reason'].value_counts().index
        ax1 = sns.countplot(data=df_ret, y='return_reason', ax=axes[1], palette="magma", order=order_reasons)
        axes[1].set_title("Головні причини повернень товарів", fontsize=14, fontweight='bold', pad=15)
        axes[1].set_xlabel("Кількість повернень", fontsize=12)
        axes[1].set_ylabel("Причина (Reason)", fontsize=12)

        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', label_type='edge', padding=5, fontweight='bold', color='#333333')

        # Немного расширяем лимит по оси X, чтобы метки цифр не вылезали за края графика
        axes[1].set_xlim(0, df_ret['return_reason'].value_counts().max() * 1.15)
    else:
        axes[1].text(0.5, 0.5, 'Немає даних для повернень', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plot_path = os.path.join(HERE, 'returns_reviews_report.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nГрафіки успішно збережено у файл: {plot_path}")
    plt.show()


if __name__ == '__main__':
    conn = get_connection()
    try:
        df_ret = analyze_returns_and_refunds(conn)
        df_rev, df_loyalty = analyze_reviews_and_loyalty(conn)
        generate_visualizations(df_ret, df_rev)
    except Exception as e:
        print(f"Виникла помилка під час аналізу: {e}")
        print("Перевірте відповідність назв колонок (наприклад, order_item_id, product_id, customer_id).")
    finally:
        conn.close()
