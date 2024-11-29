import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from flask import Flask, request, render_template, jsonify

# Load the dataset
file_path = "C:/Users/Jaeven T/Desktop/random AI stuff/aquarium rec systrem/freshwater_aquarium_fish_species.csv"
fish_data = pd.read_csv(file_path)
fish_names = fish_data['name'].tolist()

# Preprocess data for recommendations
fish_data['combined_features'] = fish_data['taxonomy'].fillna('') + ' ' + fish_data['remarks'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(fish_data['combined_features'])
similarity_matrix = cosine_similarity(feature_vectors)

# Preprocessing temperature range and pH range to extract numeric values
def extract_range(range_text):
    try:
        numbers = [float(num) for num in range_text.replace('–', '-').replace('°C', '').replace('°F', '').split('-') if num.strip().replace('.', '', 1).isdigit()]
        return sum(numbers) / len(numbers) if numbers else None
    except:
        return None

fish_data['temp_avg'] = fish_data['temprange'].apply(extract_range)
fish_data['ph_avg'] = fish_data['phRange'].apply(extract_range)
fish_data['temp_avg'] = fish_data['temp_avg'].fillna(fish_data['temp_avg'].mean())
fish_data['ph_avg'] = fish_data['ph_avg'].fillna(fish_data['ph_avg'].mean())
fish_data['combined_features'] = fish_data['taxonomy'].fillna('') + ' ' + fish_data['remarks'].fillna('')

# Vectorize and compute similarity
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(fish_data['combined_features'])
similarity_matrix = cosine_similarity(feature_vectors)

# Function to recommend fish
def recommend_fish(selected_index, top_n=5):
    similarity_scores = list(enumerate(similarity_matrix[selected_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommendations = [fish_data.iloc[i[0]]['name'] for i in sorted_scores[1:top_n+1]]
    return recommendations

def recommend_fish_with_reasoning(selected_index, top_n=5):
    similarity_scores = list(enumerate(similarity_matrix[selected_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    selected_fish = fish_data.iloc[selected_index]
    recommendations_with_reasoning = []

    for i, score in sorted_scores[1:top_n+1]:
        recommended_fish = fish_data.iloc[i]

        # Reasoning example: matching taxonomy or remarks
        reasoning = []
        if selected_fish['taxonomy'] == recommended_fish['taxonomy']:
            reasoning.append("Same taxonomy")
        if selected_fish['remarks'] and recommended_fish['remarks'] and selected_fish['remarks'] == recommended_fish['remarks']:
            reasoning.append("Similar tank requirements")
        if selected_fish['temp_avg'] and recommended_fish['temp_avg']:
            temp_difference = abs(selected_fish['temp_avg'] - recommended_fish['temp_avg'])
            if temp_difference <= 2:
                reasoning.append("Close temperature range")

        recommendations_with_reasoning.append({
            "name": recommended_fish['name'],
            "reasoning": ", ".join(reasoning) or "Similar features"
        })

    return recommendations_with_reasoning

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    selected_fish = ""
    if request.method == "POST":
        selected_fish = request.form["fish"].strip()
        if selected_fish.lower() not in [name.lower() for name in fish_names]:
            recommendations = [{"name": "Invalid fish name. Please try again.", "reasoning": ""}]
        else:
            selected_index = fish_data[fish_data['name'].str.lower() == selected_fish.lower()].index[0]
            recommendations = recommend_fish_with_reasoning(selected_index, top_n=5)
    return render_template("index.html", recommendations=recommendations, selected_fish=selected_fish)

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").lower()
    matches = [name for name in fish_names if query in name.lower()]
    return jsonify(matches)

if __name__ == "__main__":
    app.run(debug=True)