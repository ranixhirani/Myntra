import pandas as pd
import random
import numpy as np
from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load files
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")
user_data = pd.read_csv("models/user_data.csv")  # Ensure this file contains user data including body types

# Database configuration
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the pre-trained recommendation model
model = load_model('models/recommendation_model.h5')

# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    body_type = db.Column(db.String(20), nullable=False)  # Added body_type column

class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Recommendations functions
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

def content_based_recommendations(train_data, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details

def collaborative_filtering_recommendations(user_body_type, user_data, train_data, top_n=10):
    similar_users = user_data[user_data['body_type'] == user_body_type]
    if similar_users.empty:
        return pd.DataFrame()

    similar_user_ids = similar_users['user_id'].tolist()
    purchase_history = train_data[train_data['user_id'].isin(similar_user_ids)]
    
    if purchase_history.empty:
        return pd.DataFrame()

    popular_items = purchase_history['product_id'].value_counts().head(top_n).index.tolist()
    recommended_items_details = train_data[train_data['product_id'].isin(popular_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']].drop_duplicates()

    return recommended_items_details

def ai_recommendations(body_features, train_data, model, top_n=10):
    scaler = StandardScaler()
    X = train_data[['height', 'weight', 'chest', 'waist']]
    X_scaled = scaler.fit_transform(X)
    body_features_scaled = scaler.transform([body_features])
    predictions = model.predict(body_features_scaled)
    top_indices = predictions[0].argsort()[-top_n:][::-1]
    recommended_items_details = train_data.iloc[top_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details

# Routes
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]

@app.route("/")
def index():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls, random_price=random.choice(price))

@app.route("/main")
def main():
    return render_template('main.html')

@app.route("/index")
def indexredirect():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls, random_price=random.choice(price))

@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        body_type = request.form['body_type']
        new_signup = Signup(username=username, email=email, password=password, body_type=body_type)
        db.session.add(new_signup)
        db.session.commit()
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed up successfully!')

@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        new_signup = Signin(username=username, password=password)
        db.session.add(new_signup)
        db.session.commit()
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed in successfully!')

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        chest = float(request.form.get('chest'))
        waist = float(request.form.get('waist'))
        body_type = request.form.get('body_type')
        body_features = [height, weight, chest, waist]

        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)
        ai_rec = ai_recommendations(body_features, train_data, model, top_n=nbr)
        collab_rec = collaborative_filtering_recommendations(body_type, user_data, train_data, top_n=nbr)

        if content_based_rec.empty and ai_rec.empty and collab_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)
        else:
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=content_based_rec, ai_rec=ai_rec, collab_rec=collab_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls, random_price=random.choice(price))

if __name__ == '__main__':
    app.run(debug=True)
