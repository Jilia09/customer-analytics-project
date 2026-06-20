import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

conn = sqlite3.connect("online_store.db")
df_customers = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

top_cities = df_customers["city"].value_counts().head(10)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

barplot = sns.barplot(x=top_cities.values, y=top_cities.index)

plt.title("Звідки клієнти? (ТОП-10 міст)")
plt.tight_layout()
plt.show()