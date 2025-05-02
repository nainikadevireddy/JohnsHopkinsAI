from flask import Flask, request, jsonify
from datetime import datetime
import os
from glob import glob
from PIL import Image
import numpy as np
import json

from src.extraction.model import Model
from src.extraction.preprocess import Preprocessing
from src.search.indexing import KDTree
from src.search.search import KDTreeSearch, Measure
from pipeline import Pipeline

app = Flask(__name__)

# Constants
GALLERY_STORAGE = "storage/gallery"
EMBEDDINGS_STORAGE = "storage/embeddings"
ACCESS_LOGS_STORAGE = "storage/access_logs"

# Initialize components
image_size = 224
architecture = 'resnet_018'
preprocessing = Preprocessing(image_size=image_size)
model = Model(f"simclr_resources/model_size_{image_size:03}_{architecture}.pth")
index = KDTree(k=256, points=[])
search = KDTreeSearch(index, Measure.euclidean)

pipeline = Pipeline(preprocessing, model, index, search)
pipeline._precompute()

@app.route('/authenticate', methods=['POST'])
def authenticate():
    data = request.json
    image_size = data.get('image_size')
    resnet_model = data.get('resnet_model')
    
    if image_size not in [64, 224] or resnet_model not in ['resnet_018', 'resnet_034']:
        return jsonify({'status': 'error', 'message': 'Invalid model parameters'}), 400

    model_name = f"model_size_{image_size:03}_{resnet_model}"
    model_path = os.path.join("simclr_resources", f"{model_name}.pth")

    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': 'Model not found'}), 404

    try:
        model = Model(model_path)
        return jsonify({'status': 'success', 'message': 'Model selected and tree precomputed successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

    preprocessing = Preprocessing(image_size=image_size)
    model = Model(f"simclr_resources/model_size_{image_size:03}_{architecture}.pth")
    index = KDTree(k=256, points=[])
    search_euclidean = KDTreeSearch(index, Measure.euclidean)
    pipeline = Pipeline(preprocessing=preprocessing,
                        model=model,
                        index=index,
                        search = search_euclidean
                        )
    pipeline._precompute()

@app.route('/add_identity', methods=['POST'])
def add_identity():
    data = request.json
    image_path = data['image_path']
    name = data['name']
    result = pipeline.add_identity(image_path, name)
    if result:
        return jsonify({'status': 'error', 'message': result}), 400
    return jsonify({'status': 'success', 'message': 'Identity added successfully'})

@app.route('/remove_identity', methods=['DELETE'])
def remove_identity():
    data = request.json
    name = data['name']
    return jsonify({'status': 'success', 'message': 'Identity removed successfully'})

@app.route('/access_logs', methods=['GET'])
def access_logs():
    start_time_str = request.args.get('start_time')
    end_time_str = request.args.get('end_time')
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
    logs = pipeline.get_access_logs(start_time, end_time)
    return jsonify(logs)

@app.route('/search', methods=['POST'])
def search():
    file = request.files['file']
    k = int(request.form.get('k', 5))
    probe_filename = file.filename
    probe = Image.open(file).convert('RGB')
    probe = preprocessing.process(probe)
    results = pipeline.search_gallery(probe, probe_filename, k)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
