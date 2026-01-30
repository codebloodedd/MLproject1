import requests

API_KEY = "6bc7206552b8571335edd55d5e7f93a8"

url = "https://api.themoviedb.org/3/movie/popular"
params = {"api_key": API_KEY, "language": "en-US", "page": 1}

r = requests.get(url, params=params, timeout=30)
print("status:", r.status_code)
print(r.text[:400])  # prints part of response so you can see it works
r.raise_for_status()

data = r.json()
for m in data["results"][:5]:
    print(m["id"], m["title"], "| rating:", m["vote_average"], "| popularity:", m["popularity"])
