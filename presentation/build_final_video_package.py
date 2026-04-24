from pathlib import Path
import textwrap
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"
SLIDES = OUT / "slide_images"
IMG = ROOT / "website" / "assets" / "images"
IMG3 = IMG / "module3"

OUT.mkdir(exist_ok=True)
SLIDES.mkdir(exist_ok=True)

W, H = 1920, 1080
DPI = 120
BG = "#08111f"
CARD = "#101a2f"
CARD2 = "#13243d"
TEXT = "#f3f6ff"
MUTED = "#b8c7ed"
BLUE = "#7aa2ff"
GREEN = "#56d9b8"
GOLD = "#f6c85f"


def wrap(text, width=44):
    return "\n".join(textwrap.wrap(text, width=width))


def setup(title, subtitle=None, kicker=None):
    fig = plt.figure(figsize=(16, 9), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, color=BG))
    ax.add_patch(Rectangle((0, 0.72), 1, 0.28, color="#0e1c32", alpha=0.9))
    ax.add_patch(Rectangle((0, 0), 1, 0.18, color="#050b14", alpha=0.65))
    if kicker:
        ax.text(0.06, 0.91, kicker.upper(), color=GREEN, fontsize=16, weight="bold", va="top")
    ax.text(0.06, 0.84, wrap(title, 44), color=TEXT, fontsize=30, weight="bold", va="top", linespacing=1.08)
    if subtitle:
        ax.text(0.06, 0.735, wrap(subtitle, 84), color=MUTED, fontsize=15, va="top", linespacing=1.25)
    return fig, ax


def save(fig, number):
    path = SLIDES / f"slide_{number:02d}.png"
    fig.savefig(path, dpi=DPI, facecolor=BG, edgecolor=BG, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return path


def box(ax, x, y, w, h, color=CARD, edge="#263a5f", radius=0.025):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=color,
        )
    )


def bullets(ax, items, x, y, width=52, size=22, gap=0.07, color=TEXT):
    cursor = y
    for item in items:
        wrapped = wrap(item, width)
        ax.text(x, cursor, u"\u2022", color=GREEN, fontsize=size + 2, va="top", weight="bold")
        ax.text(x + 0.026, cursor, wrapped, color=color, fontsize=size, va="top", linespacing=1.18)
        cursor -= gap + 0.028 * max(0, wrapped.count("\n"))


def metric(ax, x, y, w, h, label, value, color=BLUE):
    box(ax, x, y, w, h, color="#0f2036")
    ax.text(x + 0.025, y + h - 0.045, wrap(label, 18), color=MUTED, fontsize=13, va="top")
    value_size = 24 if len(value) <= 10 else 18
    ax.text(x + 0.025, y + 0.04, wrap(value, 15), color=color, fontsize=value_size, weight="bold", va="bottom")


def image(ax, path, x, y, w, h, label=None):
    path = Path(path)
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    box_aspect = w / h
    img_aspect = iw / ih
    if img_aspect > box_aspect:
        draw_w = w
        draw_h = w / img_aspect
    else:
        draw_h = h
        draw_w = h * img_aspect
    dx = x + (w - draw_w) / 2
    dy = y + (h - draw_h) / 2
    box(ax, x, y, w, h, color="#07101d", edge="#263a5f")
    ax.imshow(im, extent=(dx, dx + draw_w, dy, dy + draw_h), zorder=3, aspect="auto")
    ax.set_aspect("auto")
    if label:
        ax.text(x + 0.015, y + 0.018, label, color=MUTED, fontsize=13, va="bottom", zorder=4)


slides = []


fig, ax = setup(
    "What Makes a Movie Successful?",
    "A TMDB movie analysis focused on attention, audience response, and practical patterns.",
    "Final video presentation",
)
ax.text(0.06, 0.58, "Core question", color=GREEN, fontsize=20, weight="bold")
ax.text(
    0.06,
    0.52,
    wrap("Why do some movies become highly visible while others earn strong approval but less attention?", 48),
    color=TEXT,
    fontsize=23,
    linespacing=1.18,
)
metric(ax, 0.06, 0.25, 0.17, 0.14, "Dataset", "TMDB")
metric(ax, 0.27, 0.25, 0.17, 0.14, "Target label", "Top 25%")
metric(ax, 0.48, 0.25, 0.17, 0.14, "Video length", "10-15")
image(ax, IMG / "04_popularity_vs_rating.png", 0.71, 0.22, 0.22, 0.34, "Popularity vs rating")
slides.append(save(fig, 1))


fig, ax = setup(
    "Introduction: Movie Success Is Not One Thing",
    "A movie can be successful by becoming widely visible, by earning strong audience approval, or by doing both.",
    "Topic",
)
bullets(
    ax,
    [
        "Popular movies are not always the highest-rated movies.",
        "Audience attention can depend on genre, release timing, budget scale, and social visibility.",
        "Ratings capture approval, while popularity captures reach and current attention.",
    ],
    0.07,
    0.61,
    width=54,
    size=24,
    gap=0.095,
)
image(ax, IMG / "10_top_genres.png", 0.58, 0.37, 0.33, 0.30, "Most common genres")
image(ax, IMG / "08_popularity_by_month.png", 0.58, 0.10, 0.33, 0.22, "Popularity by release month")
slides.append(save(fig, 2))


fig, ax = setup(
    "Research Goals and Who Benefits",
    "The project asks what we can discover and predict about movie success from observable movie attributes.",
    "Goals",
)
metric(ax, 0.07, 0.50, 0.24, 0.18, "Predict", "Top 25% movies", GREEN)
metric(ax, 0.38, 0.50, 0.24, 0.18, "Discover", "Patterns", BLUE)
metric(ax, 0.69, 0.50, 0.24, 0.18, "Explain", "Signals", GOLD)
bullets(
    ax,
    [
        "Creators and marketers can better understand which movie profiles attract attention.",
        "Streaming and recommendation teams can separate visibility signals from audience approval.",
        "Students and analysts can see how several machine-learning methods tell different parts of the same story.",
    ],
    0.10,
    0.31,
    width=78,
    size=19,
    gap=0.075,
)
slides.append(save(fig, 3))


fig, ax = setup(
    "Where the Data Came From",
    "The data came from The Movie Database (TMDB), starting with popular movie records and enriching them with details such as genres, runtime, budget, revenue, votes, and release timing.",
    "Data",
)
image(ax, IMG / "popular_raw.png", 0.06, 0.22, 0.39, 0.40, "Raw popular movie sample")
image(ax, IMG / "clean.png", 0.55, 0.22, 0.39, 0.40, "Clean modeling dataset sample")
bullets(
    ax,
    [
        "Raw data: movie records from the TMDB API.",
        "Clean data: consistent numeric fields, genre text, release timing, and success labels.",
        "Presentation focus: what the data reveals, not the cleaning process.",
    ],
    0.08,
    0.14,
    width=86,
    size=19,
    gap=0.05,
)
slides.append(save(fig, 4))


fig, ax = setup(
    "Early Patterns: Attention and Approval Differ",
    "Exploratory plots showed that popularity, rating, release timing, and genre all describe different sides of movie success.",
    "Findings",
)
image(ax, IMG / "04_popularity_vs_rating.png", 0.06, 0.32, 0.42, 0.35, "Popularity vs rating")
image(ax, IMG / "02_rating_hist.png", 0.54, 0.32, 0.38, 0.35, "Rating distribution")
bullets(
    ax,
    [
        "High visibility and high audience approval are related, but not identical.",
        "Ratings cluster in a mid-to-high range, while popularity has more extreme high-attention cases.",
        "This supports using multiple methods instead of one simple success measure.",
    ],
    0.08,
    0.18,
    width=84,
    size=20,
    gap=0.055,
)
slides.append(save(fig, 5))


fig, ax = setup(
    "Movie Profiles: PCA and Clustering",
    "PCA and clustering helped summarize the movie dataset into broad profiles without needing to judge success first.",
    "Findings",
)
image(ax, IMG / "pca_2d.png", 0.06, 0.30, 0.35, 0.37, "PCA 2D view")
image(ax, IMG / "silhouette_curve.png", 0.48, 0.30, 0.39, 0.37, "KMeans silhouette curve")
metric(ax, 0.08, 0.11, 0.23, 0.13, "2D PCA retained", "67.40%", GREEN)
metric(ax, 0.38, 0.11, 0.23, 0.13, "3D PCA retained", "82.22%", BLUE)
metric(ax, 0.68, 0.11, 0.23, 0.13, "95% variance", "5 components", GOLD)
slides.append(save(fig, 6))


fig, ax = setup(
    "Association Rules: Which Traits Appear Together?",
    "Association Rule Mining revealed recurring combinations of movie traits, such as financial scale, rating group, genre, and release timing.",
    "Findings",
)
image(ax, IMG / "arm_network_top15_lift.png", 0.06, 0.25, 0.42, 0.43, "Top lift rules network")
image(ax, IMG / "arm_support_confidence_lift.png", 0.56, 0.25, 0.34, 0.43, "Rule metrics")
bullets(
    ax,
    [
        "The strongest practical pattern was that high-budget and high-revenue movies often appeared together.",
        "Low-budget and low-revenue movies also formed a consistent pairing.",
        "These rules are useful because they are easy to explain to non-technical audiences.",
    ],
    0.08,
    0.13,
    width=84,
    size=19,
    gap=0.052,
)
slides.append(save(fig, 7))


fig, ax = setup(
    "Prediction: Which Models Found the Signal?",
    "The supervised learning goal was to predict whether a movie belongs to the top 25% of popularity.",
    "Findings",
)
image(ax, IMG3 / "nb_accuracy_comparison.png", 0.06, 0.31, 0.38, 0.36, "Naive Bayes comparison")
image(ax, IMG3 / "dt_confusion_matrix.png", 0.53, 0.31, 0.34, 0.36, "Best decision tree confusion matrix")
metric(ax, 0.08, 0.12, 0.22, 0.13, "Best NB", "0.7917", GREEN)
metric(ax, 0.39, 0.12, 0.22, 0.13, "Best tree", "0.7917", BLUE)
metric(ax, 0.70, 0.12, 0.22, 0.13, "Target", "Top 25%", GOLD)
slides.append(save(fig, 8))


fig, ax = setup(
    "Model Comparison: Best Result Was Interpretable",
    "The final comparison used Logistic Regression, Multinomial Naive Bayes, and the strongest Decision Tree.",
    "Findings",
)
image(ax, IMG3 / "regression_accuracy_comparison.png", 0.06, 0.30, 0.42, 0.38, "Model accuracy comparison")
image(ax, IMG3 / "dt_feature_importance.png", 0.56, 0.30, 0.34, 0.38, "Decision tree feature importance")
bullets(
    ax,
    [
        "Decision Tree: 0.7917 accuracy.",
        "Logistic Regression: 0.7292 accuracy.",
        "Multinomial Naive Bayes: 0.7292 accuracy.",
        "The most useful model was not only accurate; it was also easy to explain.",
    ],
    0.08,
    0.15,
    width=86,
    size=20,
    gap=0.052,
)
slides.append(save(fig, 9))


fig, ax = setup(
    "Non-Technical Conclusions",
    "The project suggests that movie success is a mixture of visibility, timing, scale, and audience response.",
    "Conclusions",
)
bullets(
    ax,
    [
        "Success is not one number. Popularity and ratings tell related but different stories.",
        "Release timing and genre shape how movies reach audiences.",
        "Financial scale matters, but it does not automatically guarantee audience approval.",
        "The most helpful findings are the ones that can be explained clearly to people making real decisions.",
    ],
    0.08,
    0.60,
    width=52,
    size=21,
    gap=0.075,
)
slides.append(save(fig, 10))


fig, ax = setup(
    "Final Takeaway",
    "A movie becomes successful through more than quality alone: attention, timing, audience behavior, and market context all work together.",
    "Wrap-up",
)
box(ax, 0.10, 0.36, 0.80, 0.22, color="#10243d")
ax.text(
    0.50,
    0.49,
    wrap("The biggest lesson: visibility and approval should be studied together because each explains a different side of movie success.", 70),
    color=TEXT,
    fontsize=24,
    ha="center",
    va="center",
    linespacing=1.18,
)
ax.text(0.50, 0.25, wrap("The project website contains the full write-up, code links, data links, visuals, and results.", 82), color=MUTED, fontsize=18, ha="center")
ax.text(0.50, 0.15, "Thank you.", color=GREEN, fontsize=36, weight="bold", ha="center")
slides.append(save(fig, 11))


script = """# Final Video Presentation Speaker Script

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
"""

(OUT / "final_video_speaker_script.md").write_text(script, encoding="utf-8")


def write_pptx(slide_paths, pptx_path):
    cx = 12192000
    cy = 6858000

    def slide_xml(idx):
        return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="{cx}" cy="{cy}"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="{cx}" cy="{cy}"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="slide_{idx:02d}.png"/>
          <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="rId1"/>
          <a:stretch><a:fillRect/></a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="{cx}" cy="{cy}"/>
          </a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''

    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(slide_paths) + 1)
    )
    content_types = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {slide_overrides}
</Types>'''

    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

    slide_ids = "\n".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, len(slide_paths) + 1)
    )
    presentation_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" saveSubsetFonts="1">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
    {slide_ids}
  </p:sldIdLst>
  <p:sldSz cx="{cx}" cy="{cy}" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>'''

    pres_rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for i in range(1, len(slide_paths) + 1):
        pres_rels.append(f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
    presentation_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {"".join(pres_rels)}
</Relationships>'''

    slide_master = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>'''

    slide_master_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>'''

    slide_layout = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>'''

    slide_layout_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>'''

    theme = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="TMDB Theme">
  <a:themeElements>
    <a:clrScheme name="TMDB"><a:dk1><a:srgbClr val="08111F"/></a:dk1><a:lt1><a:srgbClr val="F3F6FF"/></a:lt1><a:dk2><a:srgbClr val="101A2F"/></a:dk2><a:lt2><a:srgbClr val="B8C7ED"/></a:lt2><a:accent1><a:srgbClr val="7AA2FF"/></a:accent1><a:accent2><a:srgbClr val="56D9B8"/></a:accent2><a:accent3><a:srgbClr val="F6C85F"/></a:accent3><a:accent4><a:srgbClr val="13243D"/></a:accent4><a:accent5><a:srgbClr val="FFFFFF"/></a:accent5><a:accent6><a:srgbClr val="263A5F"/></a:accent6><a:hlink><a:srgbClr val="7AA2FF"/></a:hlink><a:folHlink><a:srgbClr val="7AA2FF"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>'''

    app_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application><Slides>{len(slide_paths)}</Slides><PresentationFormat>On-screen Show (16:9)</PresentationFormat>
</Properties>'''

    core_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>TMDB Movie Success Final Video Presentation</dc:title>
  <dc:creator>Pratham Tushar Shah</dc:creator>
</cp:coreProperties>'''

    if pptx_path.exists():
        pptx_path.unlink()

    with ZipFile(pptx_path, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/app.xml", app_xml)
        z.writestr("docProps/core.xml", core_xml)
        z.writestr("ppt/presentation.xml", presentation_xml)
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels)
        z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels)
        z.writestr("ppt/theme/theme1.xml", theme)
        for i, slide_path in enumerate(slide_paths, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(i))
            z.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image{i}.png"/>
</Relationships>''',
            )
            z.write(slide_path, f"ppt/media/image{i}.png")


write_pptx(slides, OUT / "tmdb_movie_success_final_video_presentation_v5.pptx")

print(f"Created {len(slides)} slide images in {SLIDES}")
print(f"Created speaker script: {OUT / 'final_video_speaker_script.md'}")
print(f"Created PowerPoint deck: {OUT / 'tmdb_movie_success_final_video_presentation_v5.pptx'}")
