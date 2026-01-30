import pandas as pd
import matplotlib.pyplot as plt

CLEAN = "data_clean/tmdb_clean.csv"

def save(name):
    plt.tight_layout()
    plt.savefig(f"images/{name}.png", dpi=200)
    plt.close()

def main():
    df = pd.read_csv(CLEAN)

    # 1) Popularity distribution
    plt.figure()
    df["popularity"].hist(bins=30)
    plt.title("Popularity Distribution")
    plt.xlabel("Popularity")
    plt.ylabel("Count")
    save("01_popularity_hist")

    # 2) Rating distribution
    plt.figure()
    df["vote_average"].hist(bins=20)
    plt.title("Rating (Vote Average) Distribution")
    plt.xlabel("Vote Average")
    plt.ylabel("Count")
    save("02_rating_hist")

    # 3) Vote count distribution (log-ish via bins)
    plt.figure()
    df["vote_count"].hist(bins=30)
    plt.title("Vote Count Distribution")
    plt.xlabel("Vote Count")
    plt.ylabel("Count")
    save("03_vote_count_hist")

    # 4) Popularity vs Rating scatter
    plt.figure()
    plt.scatter(df["vote_average"], df["popularity"], s=10)
    plt.title("Popularity vs Rating")
    plt.xlabel("Vote Average")
    plt.ylabel("Popularity")
    save("04_popularity_vs_rating")

    # 5) Runtime distribution
    plt.figure()
    df["runtime"].hist(bins=25)
    plt.title("Runtime Distribution")
    plt.xlabel("Runtime (minutes)")
    plt.ylabel("Count")
    save("05_runtime_hist")

    # 6) Budget boxplot
    plt.figure()
    plt.boxplot(df["budget"])
    plt.title("Budget Boxplot (Capped at 99th percentile)")
    plt.ylabel("Budget")
    save("06_budget_box")

    # 7) Revenue boxplot
    plt.figure()
    plt.boxplot(df["revenue"])
    plt.title("Revenue Boxplot (Capped at 99th percentile)")
    plt.ylabel("Revenue")
    save("07_revenue_box")

    # 8) Mean popularity by release month
    plt.figure()
    df.groupby("release_month")["popularity"].mean().plot(kind="bar")
    plt.title("Mean Popularity by Release Month")
    plt.xlabel("Release Month")
    plt.ylabel("Mean Popularity")
    save("08_popularity_by_month")

    # 9) Label distribution: popular_top25
    plt.figure()
    df["label_popular_top25"].value_counts().plot(kind="bar")
    plt.title("Label Distribution: Popular Top 25%")
    plt.xlabel("Label (0/1)")
    plt.ylabel("Count")
    save("09_label_popular_top25")

    # 10) Top genres by count (simple)
    # take first genre from pipe list to keep it simple and readable
    df["genre_primary"] = df["genres"].str.split("|").str[0]
    plt.figure()
    df["genre_primary"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Primary Genres (Count)")
    plt.xlabel("Primary Genre")
    plt.ylabel("Count")
    save("10_top_genres")

    print("Saved 10 plots to images/")

if __name__ == "__main__":
    main()
