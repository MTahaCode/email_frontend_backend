from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from rake_nltk import Rake
import random
import requests
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# # gemini api key to env
# gemini_api_key = os.environ["GENERATIVE_AI_API_KEY"]
# genai.configure(api_key=gemini_api_key)

# # unsplash relted
# UNSPLASH_ACCESS_KEY = os.environ["UNSPLASH_ACCESS_KEY"]
# UNSPLASH_API_URL = os.environ["UNSPLASH_API_URL"]

app = Flask(__name__)
CORS(app)

# SQLAlchemy configuration
database_uri = os.environ["DATABASE_URI"]
app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Secret Key for sessions and other purposes
app.secret_key = os.environ["SECRET_KEY"]

# Initialize SQLAlchemy and Bcrypt
db = SQLAlchemy(app)

# Define User Model
class users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Ensure the database tables are created
with app.app_context():
    db.create_all()

@app.route("/")
def index():
    return jsonify({"message": "hello from vercel"})

@app.route('/register', methods=['POST'])
def register():
    # data = request.get_json()
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    if not username or not email or not password:
        return jsonify({'error': 'Missing fields'}), 400

    # Directly use the password without hashing
    new_user = users(username=username, email=email, password=password)

    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully!'}), 201

@app.route('/login', methods=['POST'])
def login():
    # data = request.get_json()
    email = request.form.get('email')
    password = request.form.get('password')
    print(email,password)
    user = users.query.filter_by(email=email).first()
    if user and user.password == password:
        return jsonify({'message': 'Login successful!'}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

# @app.route('/query', methods=['POST'])
# def query():
#     try:
#         data = request.get_json()
#         print(data)
#         if 'query' not in data:
#             return jsonify({'error': 'No query in JSON data'}), 400

#         query_text = data['query']

#         # Define the prompt for Gemini
#         template_prompt = f"""You are a highly intelligent and professional email writer designed to understand user intent related to email text generation. The user provides you with the description of his email and you need to generate the subject, catchy promotion line and somewhat brief but persuasive description pitching the product the user has mentioned in his query.

#         Make sure that description is (max 60 words), promo is (max 15 words), and subject (max 5 words).
#         The user's query is: {query_text}

#         Respond in 3 lines, like this:
#         "subject": "[subject of the email]",
#         "promo": "[catchy one sentence/phrase promotional line]",
#         "description": "[A description of the product, make it sound like a salesman promoting and describing his product in a professional way]"
#         """

#         # Initialize the Gemini model
#         model = genai.GenerativeModel(model_name="gemini-1.5-flash")

#         # Generate content from the prompt
#         response = model.generate_content(template_prompt)
#         response_text = response.text

#         # Extract Subject, Promo, and Description from the response
#         subject_start = response_text.find('"subject"') + len('"subject"')
#         subject_end = response_text.find('"promo"')
#         promo_start = response_text.find('"promo"') + len('"promo"')
#         promo_end = response_text.find('"description"')
#         description_start = response_text.find('"description"') + len('"description"')

#         subject = response_text[subject_start:subject_end].strip().strip(':').strip().strip(',').strip()
#         promo = response_text[promo_start:promo_end].strip().strip(':').strip().strip(',').strip()
#         description = response_text[description_start:].strip().strip(':').strip().strip(',').strip()

#         # Extract keywords from the query using RAKE
#         rake = Rake()
#         rake.extract_keywords_from_text(query_text)
#         keywords = rake.get_ranked_phrases()

#         # Use the first keyword for the Unsplash API
#         if keywords:
#             keyword = keywords[0]
#             unsplash_response = requests.get(UNSPLASH_API_URL, params={
#                 'query': keyword,
#                 'client_id': UNSPLASH_ACCESS_KEY
#             })
#             unsplash_data = unsplash_response.json()
#             if 'results' in unsplash_data and len(unsplash_data['results']) > 0:
#                 image_results = unsplash_data['results']
#                 random_image = random.choice(image_results)
#                 image_url = random_image['urls']['raw']
#             else:
#                 image_url = None
#         else:
#             image_url = None

#         # Return generated subject, promo, description, and image URL
#         return jsonify({
#             'subject': subject,
#             'promo': promo,
#             'description': description,
#             'image_url': image_url
#         }), 200

#     except Exception as e:
#         print(f"Error processing Gemini's response: {e}")
#         return jsonify({'error': 'Failed to process Gemini response.'}), 500

if __name__ == "__main__":
    app.run()