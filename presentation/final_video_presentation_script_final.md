# Final Video Presentation Script

Use this script while presenting `final_video_presentation_guide.html`.

## Opening

Hello everyone. In this project, I explored a simple but important question: what makes a movie successful?

That sounds like a simple question at first, but movie success is actually more complex than it seems. A movie can be successful because it becomes highly visible, because audiences rate it strongly, or because it combines both attention and approval.

This project uses data from The Movie Database, or TMDB, to study those different sides of success together. Instead of relying on only one number, I looked at popularity, ratings, genres, release timing, financial scale, and predictive models to understand how movie success takes shape.

## 1. Movie Success Is More Than One Thing

The first important idea is that popularity and ratings do not mean the same thing.

A movie can be very popular because it gets a lot of public attention, but that does not always mean it is the highest-rated movie. At the same time, a movie can earn strong audience approval without becoming the most visible title.

That difference matters because success should not be treated as only one outcome. Looking at popularity and ratings together gives a much fuller picture of movie performance.

## 2. Why This Topic Matters

This topic matters because movies are not just creative products. They are also market products.

Their success can be shaped by genre, timing, marketing, audience expectations, and financial scale. That means success is influenced by more than quality alone.

The purpose of this project is to explain that complexity in a way that is still clear and useful.

## 3. Project Goals

The project had three main goals.

First, I wanted to predict whether a movie belongs to the top 25 percent of popularity in the dataset.

Second, I wanted to discover patterns connecting genres, timing, budget, revenue, runtime, and audience response to movie success.

Third, I wanted to explain those findings in a way that could be understood beyond a technical audience.

This kind of information could be useful for marketers, content planners, streaming platforms, and recommendation systems trying to understand what drives audience attention.

## 4. Data Source

The data came from TMDB.

The raw movie records included fields such as titles, release dates, popularity, vote averages, vote counts, and IDs. These records were then enriched with extra fields like runtime, budget, revenue, and genres.

The important point for this presentation is that the project used real movie data and produced a final dataset that could support visual analysis, pattern discovery, and supervised prediction.

## 5. First Major Finding

One of the clearest early findings was that popularity and ratings do not tell the same story.

Some movies receive high attention without being the highest-rated, while some highly rated movies are less visible. That means movie success has more than one dimension.

This is an important result because it explains why movies can succeed for different reasons. Some titles succeed through reach and visibility, while others succeed through stronger audience approval.

## 6. Movie Profiles with PCA and Clustering

Next, I used PCA and clustering to look for broader movie profiles.

In simple terms, PCA reduced the complexity of the dataset by compressing several measurements into a smaller space, and clustering grouped similar movies together.

The PCA results showed that two components retained 67.4 percent of the variation, and three components retained 82.22 percent. That tells us the dataset contains real structure. The movies do not behave like random points. They form broader numeric profiles that can be summarized meaningfully.

## 7. Association Rules: What Patterns Repeated?

The association rule results were especially useful because they revealed recurring patterns that were easy to explain.

One major pattern was that movies outside the top-popularity group were often non-summer releases. In the support table, the rule linking not-popular movies with non-summer release appeared very often, with support around 0.60.

Another strong pattern involved financial scale. High-budget movies were strongly associated with high revenue, with confidence around 0.88 and lift around 1.73. Low-budget movies were also strongly associated with low revenue, with confidence around 0.86 and lift around 1.76.

There was also evidence that low-budget movies were more likely to fall outside the top-popularity group. So the rule mining results tell a practical story: timing and financial scale were two of the most repeated patterns in the dataset.

## 8. What the Classifiers Actually Learned

For supervised learning, the target was whether a movie belonged to the top 25 percent of popularity.

The best Naive Bayes model was Bernoulli Naive Bayes, with an accuracy of 0.7917. That is important because Bernoulli Naive Bayes uses simple yes-or-no features. In other words, the model worked best when movies were described by broad category-style signals such as budget group, revenue group, release timing group, and genre membership.

That tells us something meaningful about the data. The top-popularity label was easier to separate using broad indicator patterns than by assuming smooth continuous numeric distributions.

This is also why Gaussian Naive Bayes performed so poorly, with an accuracy of 0.3125. The continuous Gaussian assumption was simply not a good fit for this task.

## 9. Decision Tree and Logistic Regression Results

The best Decision Tree reached 0.7917 accuracy, and its root split was vote_count.

That means audience engagement was the single most useful first split for separating top-popularity movies from the rest. Other trees also used release_year and release_month as their root features, which tells us that recency and timing signals were also very important.

Logistic Regression reached 0.7292 accuracy. Its strongest positive coefficients were release_year, budget, Adventure, Drama, Crime, Action, and vote_average.

In practical terms, that means newer releases, larger budgets, and certain genres were associated with a higher chance of landing in the top-popularity class.

One especially interesting result is that vote_count had a negative logistic coefficient. That suggests current popularity is not the same thing as total accumulated audience voting. A movie can have many votes overall without being in the most visible group right now.

## 10. What the Results Mean in Practice

Taken together, these results point to a specific story about movie visibility.

First, visibility is dynamic. Release year and release month mattered repeatedly, which suggests that recency and timing shape attention very strongly.

Second, scale still matters. High-budget movies were strongly linked to high revenue and were less likely to sit in the lower-visibility group.

Third, approval is different from reach. Ratings and popularity were connected, but they were not the same thing, and the models separated them in different ways.

So the overall result is not one formula for success. It is a pattern-based explanation of how timing, engagement, scale, and content traits interact.

## 11. Non-Technical Conclusions

My non-technical conclusion is that movie success is not one number.

Popularity and ratings tell related but different stories.

Release timing and audience engagement were some of the clearest signals behind visibility.

Financial scale mattered, but it did not automatically guarantee stronger audience approval.

And most importantly, the most useful findings were the ones that could explain what kind of movie tends to become visible and why.

The biggest lesson from this project is that movie success has several sides, and the best analysis explains those sides instead of collapsing them into one score.

## 12. Closing Message

To close, a movie becomes successful through more than quality alone.

Attention, timing, audience behavior, genre expectations, and market context all work together.

This project used multiple machine learning methods to study movie success from different angles. The most valuable result was not only a prediction score. It was the discovery that timing, engagement, budget, and genre each explained different pieces of how movies become visible and successful.

Thank you. The project website contains the full write-up, visuals, code links, and model results behind this presentation.
