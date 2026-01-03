import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("netflix_titles.csv")

# Drop rows with missing descriptions
df = df.dropna(subset=["description"])

print("Total titles with descriptions:", len(df))

# Description length analysis
df["description_length"] = df["description"].apply(len)

print("\nDescription length stats:")
print(df["description_length"].describe())

plt.figure()
df["description_length"].plot(kind="hist", bins=50)
plt.title("Distribution of Netflix Description Lengths")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.show()

keywords = ["love", "crime", "family", "war", "murder", "friendship"]

for word in keywords:
    count = df["description"].str.contains(word, case=False).sum()
    print(f"Titles containing '{word}': {count}")

    # Store keyword counts in a dictionary
keyword_counts = {}

for word in keywords:
    keyword_counts[word] = df["description"].str.contains(word, case=False).sum()

# Convert to DataFrame
keyword_df = pd.DataFrame(
    list(keyword_counts.items()),
    columns=["Keyword", "Count"]
)

print("\nKeyword frequency table:")
print(keyword_df)

plt.figure()
plt.bar(keyword_df["Keyword"], keyword_df["Count"])
plt.title("Common Themes in Netflix Descriptions")
plt.xlabel("Keyword")
plt.ylabel("Number of Titles")
plt.show()