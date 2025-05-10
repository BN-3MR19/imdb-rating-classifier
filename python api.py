from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

df = pd.read_csv(r"C:\Users\EGYPT\OneDrive\Documents\archive\cleaned_imdb_2024.csv")

@app.route("/movies", methods=["GET"])
def get_movies():
    movies = df["Movie_Name"].dropna().head(10).tolist()
    return jsonify({"movies": movies})

@app.route("/columns", methods=["GET"])
def get_columns():
    return jsonify({"columns": df.columns.tolist()})

@app.route("/ratings-distribution", methods=["GET"])
def rating_distribution():
    rating_counts = df['Vote_Average'].value_counts().sort_index().to_dict()
    return jsonify(rating_counts)

@app.route("/revenue-vs-rating", methods=["GET"])
def revenue_vs_rating():
    if 'US_Revenue_$' in df.columns:
        data = df[['Vote_Average', 'US_Revenue_$']].dropna().head(50)
        result = data.to_dict(orient="records")
        return jsonify(result)
    else:
        return jsonify({"error": "US_Revenue_$ column not found"}), 404

@app.route("/top-countries", methods=["GET"])
def top_countries():
    if 'Release_Country' in df.columns:
        top = df['Release_Country'].value_counts().head(10).to_dict()
        return jsonify(top)
    else:
        return jsonify({"error": "Release_Country column not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
