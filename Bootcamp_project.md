```python
!pip install kaggle
```


```python
pip install --upgrade pip
```


```python
import pandas as pd
```


```python
df = pd.read_csv('merged_data.csv', on_bad_lines='skip', engine='python')
print(f"Loaded {len(df)} rows")
```


```python
df
```


```python
print(df.shape)
print(df.columns.tolist())
df.head()
```


```python
print(df['region'].unique())
print(df['date'].min(), df['date'].max())
print(df['region'].nunique(), "regions total")
```


```python
print(df['date'].min(), df['date'].max())
```


```python
european_countries = [
    'United Kingdom', 'Germany', 'France', 'Spain', 'Italy',
    'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Finland',
    'Belgium', 'Austria', 'Switzerland', 'Poland', 'Portugal',
    'Czech Republic', 'Hungary', 'Romania', 'Greece', 'Ireland',
    'Bulgaria', 'Slovakia', 'Estonia', 'Latvia', 'Lithuania',
    'Luxembourg', 'Iceland'
]
```


```python
# Filter to European countries only
df_europe = df[df['region'].isin(european_countries)].copy()
```


```python
# Convert date to datetime
df_europe['date'] = pd.to_datetime(df_europe['date'])
```


```python
# Add a year column for grouping
df_europe['year'] = df_europe['date'].dt.year
```


```python
print(df_europe.shape)
print(df_europe['year'].value_counts().sort_index())
```


```python
# Check what's going on with 2019 and 2021
print(df_europe[df_europe['year'] == 2019][['date', 'region', 'title']].head(10))
print()
print(df_europe[df_europe['year'] == 2021][['date', 'region', 'title']].head(10))
```


```python
import os
os.makedirs('figures', exist_ok=True)
```


```python
import matplotlib.pyplot as plt #initial plot

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, col, label in zip(axes, ['af_energy', 'af_valence'], ['Energy', 'Valence (Happiness)']):
    for year, color in zip([2017, 2018, 2020], ['blue', 'green', 'red']):
        subset = df_europe[df_europe['year'] == year][col]
        subset.plot.kde(ax=ax, label=str(year), color=color)
    ax.set_title(f'{label} Distribution by Year')
    ax.legend()
    ax.set_xlabel(col)

plt.tight_layout()
plt.savefig('figures/energy_valence_distributions.png', dpi=150)
plt.show()
```


```python
# Calculate mean energy and valence by year
summary = df_europe.groupby('year')[['af_energy', 'af_valence']].mean().round(3)
print(summary)
```


```python
# Monthly trend through 2020
df_2020 = df_europe[df_europe['year'] == 2020].copy()
df_2020['month'] = df_2020['date'].dt.month

monthly = df_2020.groupby('month')[['af_energy', 'af_valence']].mean().round(3)
print(monthly)
```


```python
print(df_europe[df_europe['year'] == 2020]['date'].sort_values().unique())
```

# Pivot!!!!


```python
df_europe = df_europe[df_europe['year'].isin([2017, 2018])].copy()
print(df_europe.shape)
print(df_europe['year'].value_counts())
```


```python
# Average audio features by country
country_avg = df_europe.groupby('region')[['af_energy', 'af_valence', 'af_danceability', 'af_acousticness']].mean().round(3)
country_avg = country_avg.sort_values('af_energy', ascending=False)
print(country_avg)
```


```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

energy_order = country_avg.sort_values('af_energy', ascending=True).index

for ax, col, label, color in zip(
    axes,
    ['af_energy', 'af_valence'],
    ['Energy', 'Valence (Happiness)'],
    ['orange', 'green']
):
    ax.barh(energy_order, country_avg.loc[energy_order, col], color=color)
    ax.set_title(f'Average {label} of Top Charting Songs by European Country (2017-2018)')
    ax.set_xlabel(label)
    ax.axvline(country_avg[col].mean(), color='black', linestyle='--', label='Average')
    ax.legend()

plt.tight_layout()
plt.show()
```


```python
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(country_avg['af_energy'], country_avg['af_valence'], color='pink', s=100, zorder=3)

# Label each country dot
for country, row in country_avg.iterrows():
    ax.annotate(country, (row['af_energy'], row['af_valence']),
                textcoords="offset points", xytext=(6, 3), fontsize=10)

# Draw quadrant lines at the midpoint
ax.axhline(country_avg['af_valence'].mean(), color='gray', linestyle='--', alpha=0.5)
ax.axvline(country_avg['af_energy'].mean(), color='gray', linestyle='--', alpha=0.5)

# Label quadrants
ax.text(0.71, 0.54, 'High Energy\nHappy', fontsize=13, color='green', fontweight='bold')
ax.text(0.71, 0.43, 'High Energy\nIntense', fontsize=13, color='red', fontweight='bold')
ax.text(0.615, 0.54, 'Chill\nHappy', fontsize=13, color='blue', fontweight='bold')
ax.text(0.615, 0.43, 'Chill\nMelancholic', fontsize=13, color='purple', fontweight='bold')

ax.set_xlabel('Average Energy', fontsize=13)
ax.set_ylabel('Average Valence (Happiness)', fontsize=12)
ax.set_title('European Countries by Musical Mood (2017-2018)', fontsize=14)

plt.tight_layout()
plt.show()
```


```python
country_avg['mood_score'] = ((country_avg['af_energy'] + country_avg['af_valence']) / 2).round(3)
print(country_avg[['af_energy', 'af_valence', 'mood_score']].sort_values('mood_score', ascending=False))
```


```python
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Download world map
world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")

# Fix name mismatches
country_avg_reset = country_avg.reset_index()
country_avg_reset['region_fixed'] = country_avg_reset['region'].replace({'Czech Republic': 'Czechia'})

# Merge
europe_map = world[world['CONTINENT'] == 'Europe'].copy()
europe_map = europe_map.merge(
    country_avg_reset[['region_fixed', 'mood_score']],
    left_on='NAME', right_on='region_fixed', how='left'
)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
europe_map.plot(
    column='mood_score',
    ax=ax,
    cmap='RdYlGn',
    vmin=0.53,
    vmax=0.63,
    legend=True,
    missing_kwds={'color': 'lightgray'},
    legend_kwds={'label': 'Mood Score (Red = Dark, Green = Upbeat)', 'shrink': 0.6}
)

ax.set_xlim(-25, 45)
ax.set_ylim(34, 72)
ax.set_title('Musical Mood Score Across Europe (2017-2018)', fontsize=16)
ax.axis('off')

plt.tight_layout()

# Add country names
for idx, row in europe_map.iterrows():
    if row['geometry'] is not None and row['mood_score'] > 0:
        ax.annotate(
            row['NAME'],
            xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
            fontsize=10,
            ha='center',
            color='black'
        )
plt.show()
```


```python

```
