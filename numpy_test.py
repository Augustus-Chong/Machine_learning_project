import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")

type_count = df["Type1"].value_counts(ascending=True)

plt.barh(type_count.index, type_count.values, color="lightblue", edgecolor="black")
plt.tight_layout()
plt.show()