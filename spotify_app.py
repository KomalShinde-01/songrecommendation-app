import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set the title of the app
st.title("Spotify Mood-Based Song Recommendation")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    return data

df = load_data()

# Since we don't have a 'liked' column, let's create a dummy binary target based on popularity
# For example, liked = 1 if popularity >= 75 else 0 (you can adjust threshold)
if 'liked' not in df.columns:
    df['liked'] = (df['Popularity'] >= 75).astype(int)

# Display the dataset
st.subheader("Dataset Overview")
st.write(df.head())

# Show basic statistics
st.subheader("Statistical Summary")
st.write(df.describe())

# Visualize the distribution of liked vs disliked songs
st.subheader("Liked vs Disliked Songs")
fig, ax = plt.subplots()
sns.countplot(x='liked', data=df, ax=ax)
st.pyplot(fig)

# Feature selection for classification model
st.subheader("Feature Selection for Classification Model")

# Select only numeric columns except the target and non-feature columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
for col in ['SongID', 'liked']:
    if col in numeric_cols:
        numeric_cols.remove(col)

selected_features = st.multiselect("Select features for the model:", numeric_cols, default=numeric_cols)
# Model training
if selected_features:
    X = df[selected_features]
    y = df['liked']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    st.subheader("Model Evaluation")
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.Series(model.feature_importances_, index=selected_features)
    fig2, ax2 = plt.subplots()
    importance.sort_values().plot(kind='barh', ax=ax2)
    st.pyplot(fig2)
else:
    st.warning("Please select at least one feature to train the model.")

# Song Recommender based on mood features
st.subheader("🎵 Song Recommender Based on Your Mood")

# Sliders for features relevant to mood matching
energy = st.slider("Energy", 0.0, 1.0, 0.7)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo (BPM)", 40, 160, 120)
# We don't have valence, acousticness, instrumentalness in dataset; use Tempo instead

if st.button("Recommend a Song"):
    try:
        feature_cols = ['Energy', 'Danceability', 'TempoBPM']
        # Our dataset columns are named with uppercase initial letters, match accordingly:
        # But our sample dataset uses 'Energy', 'Danceability', 'TempoBPM' or 'Tempo'?
        # From dataset: 'Energy', 'Danceability', 'TempoBPM' (change columns accordingly)

        # Check column names in dataset
        # Let's check if column names are exactly like this:
        df.columns = [col.strip() for col in df.columns]  # Clean spaces if any

        # Adapt for exact column names
        if 'TempoBPM' in df.columns:
            feature_cols = ['Energy', 'Danceability', 'TempoBPM']
        else:
            feature_cols = ['Energy', 'Danceability', 'Tempo']

        user_input = pd.DataFrame([{
            'Energy': energy,
            'Danceability': danceability,
            'TempoBPM': tempo if 'TempoBPM' in df.columns else 0,
            'Tempo': tempo if 'Tempo' in df.columns else 0
        }])

        # Filter dataset to remove rows with missing feature values
        df_filtered = df.dropna(subset=feature_cols).copy()

        # Calculate Euclidean distance for similarity
        # Select only columns in feature_cols that exist in df
        cols = [col for col in feature_cols if col in df_filtered.columns]

        df_filtered['distance'] = ((df_filtered[cols] - user_input[cols].values) ** 2).sum(axis=1)

        top = df_filtered.sort_values(by='distance').head(1)

        st.success("Top match based on your mood:")

        # Show song title, artist, mood, and features
        display_cols = ['SongTitle', 'Artist', 'Mood'] + cols
        display_cols = [col for col in display_cols if col in df.columns]

        st.write(top[display_cols])

    except Exception as e:
        st.error(f"Error generating recommendation: {e}")
