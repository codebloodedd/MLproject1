# TMDB Movie Success - Final Video Presentation Guide

## Overview

This document is a clean speaking guide for the final video presentation. It is built from the final project website and is designed to keep the talk focused on the topic, the goals, the most interesting findings, and the non-technical conclusions.

Recommended video length: 10 to 15 minutes.

Main prediction target: top 25% popularity.

Best-performing models: Bernoulli Naive Bayes and Decision Tree.

---

## 1. Introduction to the Topic
Time: about 2 to 3 minutes

### What to say
This project explores a simple but important question: what makes a movie successful? A movie can be successful because it becomes highly visible, because viewers rate it strongly, or because it combines both attention and approval.

That difference matters because popularity and ratings do not mean the same thing. A highly popular film may be everywhere in conversation, while a highly rated film may have stronger audience approval without the same level of public attention. This project studies those different success signals together instead of treating them as one number.

### Suggested visual
`website/assets/images/04_popularity_vs_rating.png`

---

## 2. Research Goals and Who Benefits
Time: about 1 to 2 minutes

### What to say
- The first goal was to predict whether a movie belongs to the top 25 percent of popularity in the dataset.
- The second goal was to discover patterns connecting genres, release timing, runtime, budget, revenue, and audience voting behavior to movie success.
- The third goal was to explain those findings in a way a broad audience can understand.

This information could help marketers, content planners, recommendation systems, and anyone interested in understanding how movies attract attention.

### Suggested visual
`presentation/slide_images/slide_03.png`

---

## 3. Where the Data Came From
Time: about 1 minute

### What to say
The data came from The Movie Database, or TMDB. The raw data included movie records such as titles, release dates, popularity, vote averages, vote counts, and IDs. It was then enriched with more descriptive fields like runtime, revenue, budget, and genres.

For the video, keep this section simple. The audience only needs to know that the project started with real movie data, then moved into a cleaner analysis dataset. Do not spend much time on technical data preparation because the video instructions say to keep the focus on the topic itself.

### Suggested visual
`website/assets/images/clean.png`

---

## 4. Interesting Finding: Success Signals Differ
Time: about 1 to 2 minutes

### What to say
One of the first important findings was that popularity and ratings do not move together perfectly. Some movies receive high attention without being the highest-rated, while some highly rated movies are less visible. That tells us movie success should not be measured in only one way.

This finding matters because it supports the rest of the project. If success has different forms, then we need multiple methods to study it instead of relying on only one chart or one model.

### Suggested visual
`website/assets/images/02_rating_hist.png`

---

## 5. Interesting Finding: PCA and Clustering
Time: about 1 minute

### What to say
I then used PCA and clustering to look for broader movie profiles. In simple terms, PCA compresses several measurements into a smaller space so patterns are easier to see, and clustering groups similar movies together.

The PCA results showed that two components retained 67.4 percent of the variation, and three components retained 82.22 percent. This suggests there is strong structure in the movie dataset, even before applying prediction models.

### Suggested visual
`website/assets/images/pca_2d.png`

---

## 6. Interesting Finding: Association Rules
Time: about 1 to 2 minutes

### What to say
Association Rule Mining answered a different kind of question: which traits tend to appear together? This method is useful in a presentation because the results are very easy to explain.

For example, one of the strongest practical patterns was that high-budget and high-revenue movies often appeared together. Low-budget and low-revenue movies also formed a consistent pairing. This does not prove cause, but it does show that financial scale and financial outcomes are strongly connected in the dataset.

### Suggested visual
`website/assets/images/arm_network_top15_lift.png`

---

## 7. Interesting Finding: Best Prediction Results
Time: about 2 minutes

### What to say
For supervised learning, the main goal was to predict whether a movie belongs to the top 25 percent of popularity. The best Naive Bayes result came from Bernoulli Naive Bayes with an accuracy of 0.7917. The best Decision Tree also reached 0.7917.

That is interesting because both strong models were also interpretable. Bernoulli Naive Bayes worked well with clear yes-or-no style indicators, and the Decision Tree gave easy-to-understand if-then rules. This made the results useful not just for prediction, but also for explanation.

### Suggested visual
`website/assets/images/module3/regression_accuracy_comparison.png`

---

## 8. Non-Technical Conclusions
Time: about 2 minutes

### What to say
- Movie success is a combination of visibility, timing, scale, and audience response.
- Popularity and ratings should not be treated as identical.
- Release timing and genre help shape how movies reach audiences.
- Budget and revenue are connected, but financial scale alone does not explain everything.
- The most valuable findings are the ones that can be explained clearly to real decision-makers.

### Suggested visual
`website/assets/images/08_popularity_by_month.png`

---

## 9. Final Closing
Time: about 30 to 45 seconds

### What to say
The final takeaway is that a movie becomes successful through more than quality alone. Attention, timing, audience behavior, genre expectations, and market context all work together. By using multiple machine learning methods, this project looked at movie success from several angles and showed that the most useful insights are both data-supported and explainable.

Recommended ending line:

"Thank you. The project website contains the full write-up, visuals, code links, and model results behind this presentation."
