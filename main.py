import pickle
from flask import Flask
import time

MODEL_PATH = 'movie_recommender_model.pkl'
MOVIE_LIST_PATH = 'movie_list.pkl'


app = Flask(__name__)

# ----------------------------------- INIT ----------------------------------- #
# Load saved model
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

with open(MOVIE_LIST_PATH, 'rb') as f:
    movie_ids = pickle.load(f)


# ---------------------------------- ROUTES ---------------------------------- #

@app.route('/')
def home():
    return 'Hello World!'

@app.route('/recommend/<userid>')
def recommend_movies(userid):
    return recommend_movies_helper(userid, movie_ids)



# ----------------------------- HELPER FUNCTIONS ----------------------------- #

def recommend_movies_helper(user_id, list_of_movies):
    predictions = []

    for movie_id in list_of_movies:
        predicted_rating = model.predict(user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    output = ''
    for idx, ids_preds in enumerate(predictions[:20]):
        ids, preds = ids_preds
        if idx != 0: output += ','
        output += str(ids)

    return output



# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    app.run(port=8082)
