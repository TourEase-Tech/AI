from flask import Flask, jsonify, request
from ContentBased import ContentBased
from MatrixFactorization import MatrixFactorizationRecommenderSystem
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
matrix_factorization_rs = MatrixFactorizationRecommenderSystem()
content_based = ContentBased()
CORS(app)
def fit_matrix():
    print('Begin fit matrix')
    matrix_factorization_rs.fit(100, 0.01, 0.2)
    print('End fit matrix')
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(fit_matrix, 'interval', seconds = 3 * 60)
scheduler.start()

@app.route('/')
def index():
    return "Welcome to the MongoDB Server!"

@app.route('/getTours', methods = ['GET'])
def getTours():
    try:
        tours = content_based.tours
        return jsonify({'tours': tours}), 200
    except Exception as e:
        return jsonify({'error': f"An error occurred tours: {str(e)}"}), 500

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        tour_id = request.args.get('tour_id')
        if not tour_id:
            return jsonify({'error': 'Parameter tour_id is required'}), 400
        
        recommended_tours = content_based.recommend(tour_id)
        return jsonify({'recommended_tours': recommended_tours}), 200
    except Exception as e:
        return jsonify({'error': f"chiu: {str(e)}"}), 500
@app.route('/matrix_factorization', methods=['GET'])
def matrix_factorization_recommend():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'Parameter user_id is required'}), 400
        
        recommended_tours = matrix_factorization_rs.recommend_matrix_factorization(user_id)
        return jsonify({'recommended_tours': recommended_tours}), 200
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    matrix_factorization_rs.fit(5,0.01,0.2)
    app.run(debug=True)
