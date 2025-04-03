import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_select import image_select
from PIL import Image, ImageEnhance, ImageFilter
import os
import glob

nltk.download('punkt_tab')

# Define constants
DAYS = ["Day 1", "Day 2", "Day 3"]
DOMAINS = ["AI_ML", "Blockchain", "Cybersecurity", "IoT", "Web Development"]
STATES = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "West Bengal", "Uttar Pradesh", "Telangana", "Gujarat", "Rajasthan", "Punjab"]
COLLEGES = ["IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur", "IISc Bangalore", "NIT Trichy", "BITS Pilani", "VIT Vellore", "Anna University"]
FEEDBACK_OPTIONS = [
    "Amazing experience!", "Great mentors and support", "Need better resources", "Too challenging for beginners", 
    "Loved the networking opportunities", "Well-organized event", "More hands-on workshops needed", 
    "Great learning experience", "Challenging but rewarding", "Too short, need more time"
]

# New column for image filenames based on day and domain
IMAGE_FILENAMES = [f"{day}_{domain}.jpg" for day in DAYS for domain in DOMAINS]

# Function to generate dataset
def generate_dataset(num_rows):
    X, _ = make_classification(n_samples=num_rows, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    
    data = {
        "Participant_ID": [f"P{str(i).zfill(3)}" for i in range(1, num_rows + 1)],
        "Domain": [random.choice(DOMAINS) for _ in range(num_rows)],
        "Day": [random.choice(DAYS) for _ in range(num_rows)],
        "College": [random.choice(COLLEGES) for _ in range(num_rows)],
        "State": [random.choice(STATES) for _ in range(num_rows)],
        "Experience_Level": [random.choice(["Beginner", "Intermediate", "Advanced"]) for _ in range(num_rows)],
        "Team_Size": np.clip((X[:, 0] * 2 + 3).astype(int), 2, 6),
        "Project_Score": np.clip((X[:, 1] * 15 + 75).astype(int), 50, 100),
        "Completion_Status": [random.choice(["Completed", "Incomplete"]) for _ in range(num_rows)],
        "Feedback": [random.choice(FEEDBACK_OPTIONS) for _ in range(num_rows)],
        "Image_Filename": [random.choice(IMAGE_FILENAMES) for _ in range(num_rows)],
    }
    return pd.DataFrame(data)

# Streamlit app
st.title("Hackathon Event Analysis")

# Sidebar for selecting tasks
st.sidebar.title("Select a Task")
task = st.sidebar.radio("Choose an option:", ["Dataset Generation", "Dashboard Development", "Text Analysis", "Image Processing"])

if task == "Dataset Generation":
    st.subheader("Generate Hackathon Dataset")
    num_rows = st.number_input("Enter number of rows to generate", min_value=100, max_value=1000, value=350, step=50)
    if st.button("Generate Dataset"):
        df = generate_dataset(num_rows)
        st.write("### Preview of Generated Dataset")
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Dataset as CSV", data=csv, file_name="hackathon_dataset.csv", mime="text/csv")

elif task == "Dashboard Development":

    # Load dataset
    @st.cache_data
    def load_data():
        return generate_dataset(350)

    df = load_data()

    # Sidebar filters
    st.sidebar.subheader("Filter Data")
    selected_domain = st.sidebar.selectbox("Select Domain", ["All"] + list(df["Domain"].unique()))
    selected_state = st.sidebar.selectbox("Select State", ["All"] + list(df["State"].unique()))
    selected_college = st.sidebar.selectbox("Select College", ["All"] + list(df["College"].unique()))

    # Apply filters
    filtered_df = df.copy()
    if selected_domain != "All":
        filtered_df = filtered_df[filtered_df["Domain"] == selected_domain]
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["State"] == selected_state]
    if selected_college != "All":
        filtered_df = filtered_df[filtered_df["College"] == selected_college]

    st.subheader("Hackathon Participation Dashboard")

    # 1. Bar Chart - Participants per Domain
    st.write("### Participants per Domain")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="Domain", palette="viridis", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 2. Pie Chart - Participants per Day
    st.write("### Participants per Day")
    fig, ax = plt.subplots()
    filtered_df["Day"].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("pastel"), ax=ax)
    st.pyplot(fig)

    # 3. Bar Chart - Participants per State
    st.write("### Participants per State")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="State", palette="coolwarm", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 4. Pie Chart - Completion Status
    st.write("### Completion Status")
    fig, ax = plt.subplots()
    filtered_df["Completion_Status"].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("Set2"), ax=ax)
    st.pyplot(fig)

    # 5. Bar Chart - Average Project Score by Domain
    st.write("### Average Project Score by Domain")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="Domain", y="Project_Score", palette="magma", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif task == "Text Analysis":
    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Load dataset
    @st.cache_data
    def load_data():
        return generate_dataset(350)

    df = load_data()

    # Preprocess feedback
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())  # Tokenization & Lowercasing
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]  # Stopword removal & Lemmatization
        return " ".join(filtered_tokens)

    df["Lemmatized_Feedback"] = df["Feedback"].apply(preprocess_text)

    # Sidebar for domain selection
    selected_domain = st.sidebar.selectbox("Select Domain for Word Cloud", df["Domain"].unique())

    # Generate Word Cloud
    st.subheader("Word Cloud for Domain-wise Feedback")
    domain_feedback = " ".join(df[df["Domain"] == selected_domain]["Lemmatized_Feedback"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(domain_feedback)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Compute TF-IDF and Cosine Similarity
    st.subheader("Feedback Similarity Analysis")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Lemmatized_Feedback"])

    cosine_sim = cosine_similarity(tfidf_matrix)

    # Display Heatmap of Similarity Scores
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cosine_sim[:20, :20], cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("Above is a heatmap displaying cosine similarity scores for participant feedback. Darker shades indicate higher similarity.")

elif task == "Image Processing":
    # Define image folder path
    IMAGE_FOLDER = "images"  # Ensure this folder is in the same path as app.py

    # Get all image file paths
    image_files = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))
    print(image_files)
    image_dict = {os.path.basename(img): img for img in image_files}

    # Sidebar filter by day
    st.sidebar.subheader("Filter by Day")
    selected_day = st.sidebar.selectbox("Select Day", ["All"] + ["Day 1", "Day 2", "Day 3"])

    # Filter images based on selection
    filtered_images = [img for img in image_dict if selected_day in img] if selected_day != "All" else list(image_dict.keys())

    st.subheader("Day-wise Image Gallery")

    # Display gallery with image_select widget
    if filtered_images:
        selected_image_name = image_select("Select an image", filtered_images)
    else:
        st.write("No images found for the selected day.")
        selected_image_name = None

    # Image Processing Section
    if selected_image_name:
        image_path = image_dict[selected_image_name]
        image = Image.open(image_path).convert("RGB")
        
        # Image processing sliders
        st.subheader("Image Processing")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
        sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0)
        blur = st.slider("Blur", 0.0, 5.0, 0.0)

        # Apply real-time image processing
        processed_image = ImageEnhance.Brightness(image).enhance(brightness)
        processed_image = ImageEnhance.Sharpness(processed_image).enhance(sharpness)
        if blur > 0:
            processed_image = processed_image.filter(ImageFilter.GaussianBlur(blur))

        # Display processed image
        st.image(processed_image, caption="Processed Image", use_container_width=True)

        # Download processed image
        processed_image_path = f"processed_{selected_image_name}"
        processed_image.save(processed_image_path)
        with open(processed_image_path, "rb") as file:
            st.download_button("Download Processed Image", file, file_name=selected_image_name, mime="image/jpeg")
