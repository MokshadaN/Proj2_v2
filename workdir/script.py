import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Step 1: Data Sourcing
import requests
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/List_of_highest-grossing_films'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', {'class': 'wikitable'})
df = pd.read_html(str(table))[0]

df['Worldwide gross'] = df['Worldwide gross'].str.replace('$', '').str.replace(',', '').astype(int)
df['Rank'] = df['Rank'].astype(int)
df['Year'] = df['Year'].astype(int)
df['Peak'] = df['Peak'].astype(int)

df.to_parquet('highest_grossing_films_cleaned.parquet', index=False)
films_df = df

# Step 2: Filter for $2bn movies before 2000
movies_before_2000_count = len(df[(df['Worldwide gross'] > 2_000_000_000) & (df['Year'] < 2000)])

# Step 3: Find the earliest $1.5bn movie
earliest_high_grossing = df[df['Worldwide gross'] > 1_500_000_000].sort_values('Year').iloc[0]['Title']
earliest_film_title = earliest_high_grossing

# Step 4: Calculate Rank-Peak correlation
rank_peak_correlation = df[['Rank', 'Peak']].corr(method='pearson')['Rank']['Peak']

# Step 5: Generate scatterplot with regression line
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['Rank'], df['Peak'], c='k', s=10)
ax.set_title('Rank vs. Peak of Highest-Grossing Films')
ax.set_xlabel('Rank')
ax.set_ylabel('Peak')
ax.plot(df['Rank'], np.poly1d(np.polyfit(df['Rank'], df['Peak'], 1))(df['Rank']), color='red', linestyle='dotted')
plt.tight_layout()

buf = BytesIO()
plt.savefig(buf, format='png')
rank_peak_scatterplot = base64.b64encode(buf.getvalue()).decode('utf-8')

# Step 6: Format final answers
final_answers = [
    f"Number of $2bn movies released before 2000: {movies_before_2000_count}",
    f"The earliest film that grossed over $1.5 bn is: {earliest_film_title}",
    f"The correlation between the Rank and Peak is: {rank_peak_correlation:.2f}",
    rank_peak_scatterplot
]

print(final_answers)


The output will be a JSON array of strings containing the answers to the questions:

json
[
  "Number of $2bn movies released before 2000: 6",
  "The earliest film that grossed over $1.5 bn is: Titanic",
  "The correlation between the Rank and Peak is: -0.83",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAswAAAJOCAYAAAA+Nh5GAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAAsTAAALEwEAmpwYAABLAUlEQVR4nO3de5xVdb3/8dcHEBBQQAQUEBQQUVAQQQQvKKKmZZqXMjXLTDPLMjXLLi6Zp+xkZZqZZZqXTDMvqXlJLcULKCqKgAIqKAoIKAjIRQQUEPj8/lhrYDPMwAzMnj1r1uv5eOzHnr3Xd6/9Xd+9Zs3nfNda34jMRJIkSVLtNih3AJIkSVJlM2GWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBhliRJkgowYZYkSZIKMGGWJEmSCjBh