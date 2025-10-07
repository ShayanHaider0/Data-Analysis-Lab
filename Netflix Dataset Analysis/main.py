# Netflix Dataset Analysis with Pandas + Matplotlib (only pyplot)

# ------------------------
# 0. Import Libraries
# ------------------------
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# 1. Load Dataset
# ------------------------
df = pd.read_csv("netflix_titles.csv")

# ------------------------
# 2. Data Cleaning
# ------------------------
df['director'] = df['director'].fillna("Unknown")
df['cast'] = df['cast'].fillna("Unknown")
df['country'] = df['country'].fillna("Unknown")

# Clean date_added (remove extra spaces, parse safely)
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors="coerce")

# Extract year and month
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# ------------------------
# 3. Analysis & Visualizations
# ------------------------

# (a) Movies vs TV Shows
type_counts = df['type'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(type_counts, labels=type_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Movies vs TV Shows on Netflix")
plt.show()

# (b) Content release trend by year
year_counts = df['release_year'].value_counts().sort_index()

plt.figure(figsize=(12,6))
plt.plot(year_counts.index, year_counts.values, marker='o')
plt.title("Number of Titles Released by Year")
plt.xlabel("Year")
plt.ylabel("Count of Titles")
plt.grid(True)
plt.show()

# (c) Most common genres
df_genres = df.assign(genre=df['listed_in'].str.split(',')).explode('genre')
df_genres['genre'] = df_genres['genre'].str.strip()

top_genres = df_genres['genre'].value_counts().head(10)

plt.figure(figsize=(10,6))
plt.barh(top_genres.index[::-1], top_genres.values[::-1])  # reversed for better order
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.show()

# (d) Ratings distribution
rating_counts = df['rating'].value_counts()

plt.figure(figsize=(10,6))
plt.barh(rating_counts.index[::-1], rating_counts.values[::-1])
plt.title("Distribution of Content Ratings")
plt.xlabel("Count")
plt.ylabel("Rating")
plt.show()

# ------------------------
# 4. Key Insights
# ------------------------
print("✅ Netflix has more", df['type'].value_counts().idxmax(), "than", df['type'].value_counts().idxmin())
print("✅ Content peaked around:", df['release_year'].value_counts().idxmax())
print("✅ Top Genre:", top_genres.index[0])
print("✅ Most targeted rating:", df['rating'].value_counts().idxmax())
