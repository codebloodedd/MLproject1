import requests
import pandas as pd
import time

API_KEY = "6bc7206552b8571335edd55d5e7f93a8"
BASE = "https://api.themoviedb.org/3"

def get_popular_pages(pages=10):
    rows = []
    for page in range(1, pages + 1):
        r = requests.get(
            f"{BASE}/movie/popular",
            params={"api_key": API_KEY, "language": "en-US", "page": page},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        rows.extend(data["results"])
        time.sleep(0.25)
    return pd.DataFrame(rows)

def enrich_movie_details(movie_id):
    r = requests.get(
        f"{BASE}/movie/{movie_id}",
        params={"api_key": API_KEY, "language": "en-US"},
        timeout=30
    )
    r.raise_for_status()
    return r.json()

def main():
    # 10 pages ~ 200 movies (enough for Module 1, scalable later)
    df = get_popular_pages(pages=10)
    df.to_csv("data_raw/tmdb_popular_raw.csv", index=False)
    print("Saved raw list:", df.shape)

    # Enrich with budget/revenue/runtime/genres
    details = []
    for mid in df["id"].tolist():
        try:
            d = enrich_movie_details(mid)
            details.append({
                "id": d.get("id"),
                "runtime": d.get("runtime"),
                "budget": d.get("budget"),
                "revenue": d.get("revenue"),
                "genres": "|".join([g["name"] for g in d.get("genres", [])]),
                "release_date_full": d.get("release_date"),
                "original_language": d.get("original_language"),
                "status": d.get("status")
            })
            time.sleep(0.25)
        except Exception as e:
            print("Failed id", mid, e)

    df_details = pd.DataFrame(details)
    df2 = df.merge(df_details, on="id", how="left")
    df2.to_csv("data_raw/tmdb_enriched_raw.csv", index=False)
    print("Saved enriched raw:", df2.shape)
    print(df2.head(3))

if __name__ == "__main__":
    main()
