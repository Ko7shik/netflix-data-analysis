import pandas as pd

df = pd.read_csv("netflix_titles.csv")

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nCount of Movies vs TV Shows:")
print(df['type'].value_counts())

print("\nTop 10 countries:")
print(df['country'].value_counts().head(10))

print("\nContent per release year:")
print(df['release_year'].value_counts().head(10))

# Fill missing countries with 'Unknown'
df['country'] = df['country'].fillna('Unknown')

# Drop rows where title is missing (important)
df = df.dropna(subset=['title'])

print("\nAfter cleaning:")
print(df.isnull().sum())

# Movies vs TV Shows per year
movies_per_year = df[df['type'] == 'Movie']['release_year'].value_counts().sort_index()
tv_per_year = df[df['type'] == 'TV Show']['release_year'].value_counts().sort_index()

print("\nMovies released per year:")
print(movies_per_year.tail(10))

print("\nTV Shows released per year:")
print(tv_per_year.tail(10))

import matplotlib.pyplot as plt

# BAR CHART
type_counts = df['type'].value_counts()

plt.figure()
plt.bar(type_counts.index, type_counts.values)
plt.title("Movies vs TV Shows on Netflix")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()   # <-- FIRST SHOW


# LINE GRAPH
year_counts = df['release_year'].value_counts().sort_index()

plt.figure()
plt.plot(year_counts.index, year_counts.values)
plt.title("Netflix Content Growth Over Years")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.show()   # <-- SECOND SHOW

# 1. Movie or TV Show as binary
df['is_movie'] = df['type'].apply(lambda x: 1 if x == 'Movie' else 0)

# 2. Description length (text feature)
df['description_length'] = df['description'].apply(len)

print("\nFeature Engineered Columns:")
print(df[['type', 'is_movie', 'description_length']].head())

import matplotlib.pyplot as plt

yearly_counts = df['release_year'].value_counts().sort_index()
recent_growth = yearly_counts[yearly_counts.index >= 2010]

plt.figure()
recent_growth.plot()
plt.title("Netflix Content Growth Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.show()