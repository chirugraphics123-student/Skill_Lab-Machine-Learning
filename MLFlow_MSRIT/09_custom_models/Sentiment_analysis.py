import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mlflow.pyfunc import PythonModel

# ─── 1. Dataset ───────────────────────────────────────────────────
data = pd.DataFrame({
    "text": [
        # POSITIVE
        "An incredible masterpiece with stunning visuals and a gripping plot",
        "The acting was phenomenal and the story kept me on the edge of my seat",
        "A heartwarming film with beautiful cinematography and memorable characters",
        "Brilliant direction, outstanding performances, and a perfect ending",
        "One of the best movies I have ever seen, absolutely loved every minute",
        "A truly moving experience that left me in tears and smiling at the same time",
        "The screenplay is tight, the cast is phenomenal, and the score is unforgettable",
        "I walked out of the theater feeling inspired and emotionally fulfilled",
        "Every scene is perfectly crafted, the pacing is flawless, the cast is superb",
        "A dazzling achievement that redefines what cinema can be",
        "The chemistry between the leads was electric and the dialogue was razor sharp",
        "I have already rewatched it twice and it only gets better each time",
        "A bold and ambitious film that delivers on every front",
        "The twist at the end was brilliant and the whole build up was perfectly executed",
        "Gorgeous to look at, deeply felt, and endlessly rewatchable",
        "This is the kind of filmmaking that makes you believe in the magic of cinema",
        "The performances are so natural you forget you are watching actors",
        "A powerful story told with grace, humor, and emotional depth",
        "I was completely hooked from the first scene to the last",
        "The direction is masterful and the emotional payoff is absolutely worth it",
        # NEGATIVE
        "A boring and predictable waste of time with terrible dialogue",
        "The plot made no sense and the acting was painfully wooden",
        "A disappointing mess with bad pacing and unlikable characters",
        "Worst movie of the year, I want my money back",
        "The story was incoherent and the direction was completely flat",
        "I fell asleep twice and had to force myself to finish watching",
        "Every single character is annoying and the plot holes are enormous",
        "It feels like a first draft that somehow made it to the big screen",
        "The acting is so bad it makes me wince every time someone opens their mouth",
        "A two hour ordeal that I will never be able to get back",
        "The script reads like it was written by someone who has never seen a movie",
        "I kept checking my phone every five minutes hoping the credits would roll",
        "The villain is one dimensional, the hero is boring, and the twist is obvious",
        "It tries to be deep but ends up being shallow and pretentious",
        "The CGI is terrible and the action scenes are impossible to follow",
        "I have seen student films with better storytelling and production values",
        "The soundtrack is annoying, the pacing is glacial, and the ending is insulting",
        "Every joke falls flat and every emotional beat feels forced",
        "It is an insult to the audience to think they will sit through this",
        "I left the theater before the movie was over and I regret wasting even that time",
        # NEUTRAL
        "A decent movie, nothing special but not bad either",
        "The film was okay, it had some good parts and some weak ones",
        "An average movie that you might watch once and forget",
        "The movie was fine, neither great nor terrible",
        "A middling film with a few highlights but mostly forgettable",
        "It is the kind of movie you watch on a lazy Sunday and move on with your life",
        "The first half was promising but it loses its way in the second act",
        "A competent but unremarkable effort that neither excites nor offends",
        "The cast tries their best but the material does not give them much to work with",
        "It is a safe, formulaic film that will please some and bore others",
        "The visuals are nice but the story does not do much with them",
        "A passable entertainment option if you cannot think of anything better to do",
        "The movie has its moments but never quite commits to a tone",
        "It is neither a waste of time nor a must watch, just a middle of the road film",
        "The direction is competent but unoriginal and the script plays it too safe",
        "I would not recommend it as a must see but I also would not warn you away",
        "A forgettable entry in the genre that does not break any new ground",
        "The film is adequate but leaves you wanting more after the credits roll",
        "It is a fine enough movie for a casual viewing but not one I will revisit",
        "The movie does what it sets out to do, which is not very much",
    ],
    "sentiment": ["positive"] * 20 + ["negative"] * 20 + ["neutral"] * 20,
})

# ─── 2. Train / test split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["sentiment"],
    test_size=0.3, random_state=42, stratify=data["sentiment"],
)

# ─── 3. Build pipeline ────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
    )),
])

# ─── 4. Train ─────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

# ─── 5. Evaluate ──────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# ─── 6. Save pipeline locally ─────────────────────────────────────
artifact_dir = Path("model_artifacts")
artifact_dir.mkdir(exist_ok=True)
model_path = artifact_dir / "movie_sentiment.joblib"
joblib.dump(pipeline, model_path)

# ─── 7. MLflow custom model wrapper ───────────────────────────────
class MovieSentimentModel(PythonModel):
    def load_context(self, context):
        self.classifier = joblib.load(context.artifacts["classifier"])

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str)
        labels = self.classifier.predict(texts)
        probabilities = self.classifier.predict_proba(texts)
        confidence = probabilities.max(axis=1)
        return pd.DataFrame({
            "text": texts.tolist(),
            "predicted_sentiment": labels,
            "confidence": confidence.round(4),
        })

# ─── 8. Log to MLflow ─────────────────────────────────────────────
example = pd.DataFrame({
    "text": [
        "This movie was absolutely amazing",
        "A complete waste of time",
        "It was okay, nothing special",
    ]
})

mlflow.set_experiment("Movie Sentiment - 3 Class")

with mlflow.start_run(run_name="tfidf-lr-movie-sentiment"):
    # Log metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("vectorizer", "TfidfVectorizer ngram(1,2)")
    mlflow.log_param("n_samples", len(data))
    mlflow.log_metric("accuracy", (y_pred == y_test).mean())

    # Log the custom pyfunc model
    model_info = mlflow.pyfunc.log_model(
        name="movie_sentiment_model",
        python_model=MovieSentimentModel(),
        artifacts={"classifier": str(model_path.resolve())},
        input_example=example,
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            f"pandas=={pd.__version__}",
            f"scikit-learn=={__import__('sklearn').__version__}",
            f"joblib=={joblib.__version__}",
        ],
    )

    print(f"\nModel URI: {model_info.model_uri}")

# ─── 9. Load & predict from MLflow ────────────────────────────────
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
new_reviews = pd.DataFrame({"text": [
    "A stunning and beautiful film",
    "Terrible and boring, waste of time",
    "The movie was fine, nothing remarkable",
]})
print("\n── Predictions ──")
print(loaded_model.predict(new_reviews))
