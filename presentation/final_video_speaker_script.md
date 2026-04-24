# Final Video Presentation Speaker Script

Target time: about 13-14 minutes.

## Slide 1 - Title: What Makes a Movie Successful? (about 45 seconds)

Hello, today I am presenting my machine learning project on movie success using data from The Movie Database, also known as TMDB. The main question I wanted to explore is: what makes a movie successful? I looked at success from more than one angle, because a movie can be successful by becoming highly visible, by earning strong audience approval, or by doing both. Throughout the project website, I used exploratory analysis, pattern-finding methods, and supervised prediction models to understand which movie attributes seem connected to popularity and audience response.

## Slide 2 - Introduction: Movie Success Is Not One Thing (about 1 minute 30 seconds)

To start, I do not want to assume that the viewer already knows the topic. Movies are cultural products, but they are also business products. They are shaped by genre, release timing, audience expectations, marketing, and how much attention they receive after release. One important idea in this project is that popularity and ratings are not the same thing. A movie can be very popular because many people are talking about it, but it might not be the highest-rated movie. Another movie might have strong ratings but less mainstream visibility. That is why this topic is interesting: success is not just one number.

## Slide 3 - Research Goals and Who Benefits (about 1 minute 20 seconds)

The project had three main goals. First, I wanted to predict whether a movie falls into the top 25 percent of popularity in the dataset. Second, I wanted to discover patterns among movie features such as genres, revenue, budget, runtime, release timing, and audience voting behavior. Third, I wanted to explain the findings in a way that would be understandable to a broad audience. This kind of information could benefit movie marketers, streaming platforms, recommendation teams, and anyone trying to understand how movie traits connect to audience attention.

## Slide 4 - Where the Data Came From (about 1 minute)

The data came from the TMDB API. The raw data included movie records such as titles, release dates, popularity, vote averages, vote counts, and IDs. The data was then enriched with additional details like runtime, budget, revenue, and genres. On this slide, the left side shows a raw data preview and the right side shows the cleaner version used for analysis. I will not spend time on cleaning details, because the goal of this presentation is the topic and findings. The important point is that the final dataset had consistent columns that could support visual exploration and modeling.

## Slide 5 - Early Patterns: Attention and Approval Differ (about 1 minute 20 seconds)

One of the first interesting findings was that popularity and ratings do not move together perfectly. The scatterplot shows that some movies receive high attention without being the highest-rated, while some highly-rated movies are not the most popular. This supports the idea that visibility and audience approval are related but different. The rating distribution also shows that many movies cluster in a middle-to-high rating range. That means ratings alone may not be enough to explain why a movie becomes highly visible.

## Slide 6 - Movie Profiles: PCA and Clustering (about 1 minute 15 seconds)

Next, I used dimensionality reduction and clustering to think about movie profiles. In plain language, this helped compress several movie measurements into a smaller visual space so that patterns could be easier to see. The PCA results showed that two components retained about 67 percent of the variation, and three components retained about 82 percent. Five components were needed to retain at least 95 percent. The clustering work suggested that movies do form broad groups based on their numeric profiles, but those groups are not always perfectly separated.

## Slide 7 - Association Rules: Which Traits Appear Together? (about 1 minute 20 seconds)

Association Rule Mining helped answer a different kind of question: which movie traits tend to appear together? This is useful because the output is very explainable. For example, one of the strongest practical patterns was that high-budget and high-revenue movies appeared together often. The opposite pairing, low-budget and low-revenue, also appeared consistently. This does not mean budget causes success by itself, but it shows that financial scale and revenue signals are strongly connected in the dataset. These rules are useful because they are easy to describe without technical language.

## Slide 8 - Prediction: Which Models Found the Signal? (about 1 minute 40 seconds)

For supervised prediction, the main target was whether a movie belonged to the top 25 percent of popularity. I compared different Naive Bayes versions and decision tree models. The best Naive Bayes version was Bernoulli Naive Bayes, with an accuracy of 0.7917. This suggests that simple yes-or-no indicators, such as whether a movie has a certain genre or falls into a high-budget or high-revenue group, worked well for this dataset. The best decision tree also reached 0.7917 accuracy. The value of the tree is that it gives if-then style rules, which are easier to explain than many other models.

## Slide 9 - Model Comparison: Best Result Was Interpretable (about 1 minute 20 seconds)

For the final comparison, I compared the Decision Tree, Logistic Regression, and Multinomial Naive Bayes on the same binary target. The Decision Tree performed best, with 0.7917 accuracy. Logistic Regression and Multinomial Naive Bayes both reached 0.7292 accuracy. The key result is not only that the decision tree performed best, but that it was also interpretable. For a topic like movie success, interpretability matters because the goal is not only to make predictions, but also to explain what kinds of signals seem connected to popularity.

## Slide 10 - Non-Technical Conclusions (about 1 minute 50 seconds)

My non-technical conclusion is that movie success is a combination of visibility, timing, scale, and audience response. Popularity and ratings should not be treated as identical. Release timing and genre help shape how movies reach audiences. Budget and revenue are connected, but financial scale alone does not tell the full story of audience approval. The most useful findings from this project are the ones that can be explained clearly: certain movie profiles tend to attract more attention, and some features help separate top-popularity movies from the rest. In a real-world setting, these insights could support marketing strategy, recommendation design, and content planning.

## Slide 11 - Final Takeaway (about 45 seconds)

The final takeaway is that a movie becomes successful through more than quality alone. Attention, timing, audience behavior, genre expectations, and market context all work together. This project used several machine learning methods to study those patterns from different angles, and the project website contains the full write-up, visuals, code links, data links, and model results. Thank you.
