from flask import Flask, request, jsonify, send_from_directory
from src.inference import object_detection, non_maximal_suppression, video_processing
from pipeline import InferenceService
from src.rectification import hard_negative_mining
import src.metrics as metrics
import os

app = Flask(__name__)

# Global variables
inference_service = None
hard_negative_miner = None
stream = None


@app.route('/')
def index():
    """
    Index route that returns a welcome message.

    Returns:
    JSON response with a welcome message.
    """
    return "Welcome to the Object Detection System"



@app.route('/process_video', methods=['POST'])
def process_video():
    global inference_service, stream
    annotation_format = request.json.get('format', 'YOLO').upper()

    if annotation_format not in ['YOLO', 'PASCAL', 'COCO']:
        return jsonify({"error": "Invalid format specified"}), 400

    inference_service.output_format = annotation_format

    print("HERE")
    frame_counter = 0
    for frame in stream.capture_udp_stream():
        print(frame_counter)
        detections = inference_service.detect(frame)
        print(detections)
        if detections:
            inference_service._save(frame, detections)
        frame_counter += 1

    return jsonify({"status": "Processing completed", "total_frames": frame_counter})


@app.route('/list_detections', methods=['GET'])
def list_detections():
    frame_range = request.args.get('frame_range', default="0-100", type=str)
    start_frame, end_frame = map(int, frame_range.split('-'))
    results = []

    for i, detections in enumerate(inference_service.detect()):
        if start_frame <= i <= end_frame:
            results.append(detections)

    return jsonify(results)

@app.route('/top_hard_negatives', methods=['GET'])
def top_hard_negatives():
    num_hard_negatives = request.args.get('top_n', default=10, type=int)
    hard_negatives = hard_negative_miner.sample_hard_negatives(num_hard_negatives, criteria='total_loss')
    base_names = hard_negatives[['image_file', 'annotation_file']].applymap(os.path.basename).values.tolist()
    save_dir = 'augmented_hard_negatives'
    hard_negative_miner.augment_and_save(hard_negatives, save_dir)
    return jsonify(base_names)

@app.route('/images_with_detections', methods=['GET'])
def images_with_detections():
    frame_range = request.args.get('frame_range', default="0-100", type=str)
    start_frame, end_frame = map(int, frame_range.split('-'))
    annotation_format = request.args.get('format', default='YOLO', type=str).upper()

    if annotation_format not in ['YOLO', 'PASCAL', 'COCO']:
        return jsonify({"error": "Invalid format specified"}), 400

    inference_service.output_format = annotation_format
    save_dir = 'detections_with_annotations'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = []
    for i, detections in enumerate(inference_service.detect()):
        if start_frame <= i <= end_frame:
            frame_filename = os.path.join(save_dir, f'frame_{i}.jpg')
            inference_service._save(frame_filename, detections)
            results.append({
                'image': f'frame_{i}.jpg',
                'annotations': frame_filename.replace('.jpg', f'.{annotation_format.lower()}')
            })

    return jsonify(results)


@app.route('/detections/<filename>')
def get_detection_file(filename):
    return send_from_directory('storages/prediction', filename)

def initialize_services():
    global inference_service, hard_negative_miner, stream

    cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_1.cfg"
    weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_1.weights"
    names_path = "yolo_resources/logistics.names"
    score_threshold = .5
    iou_threshold = .4

    yolo_detector = object_detection.YOLOObjectDetector(cfg_path, weights_path, names_path)
    nms = non_maximal_suppression.NMS(score_threshold, iou_threshold)

    video_source = 'udp://127.0.0.1:23000?overrun_nonfatal=1&fifo_size=50000000'
    stream = video_processing.VideoProcessing(video_source)

    inference_service = InferenceService(stream, yolo_detector, nms, output_format='YOLO')

    dataset_dir = 'logistics'  # Adjust the path as necessary
    loss_parameters = {'num_classes': 20, 'lambda_coord': 5., 'lambda_noobj': 1.}
    hard_negative_miner = hard_negative_mining.HardNegativeMiner(
        model=yolo_detector,
        nms=nms,
        measure=metrics.Loss(**loss_parameters),
        dataset_dir=dataset_dir
    )
    hard_negative_miner.extract_zip('logistics.zip', 'logistics')

if __name__ == "__main__":
    initialize_services()
    app.run(host='0.0.0.0', port=5001)
