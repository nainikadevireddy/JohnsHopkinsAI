import os
from glob import glob

from src.extraction.model import Model
from src.extraction.preprocess import Preprocessing
from src.search.indexing import KDTree
from src.search.search import KDTreeSearch, Measure

from PIL import Image
import numpy as np


# Location of storage
GALLERY_STORAGE = "storage/gallery"
EMBEDDINGS_STORAGE = "storage/embeddings"
ACCESS_LOGS_STORAGE = "storage/access_logs"


class Pipeline:

    def __init__(self, preprocessing, model, index, search):
        self.preprocessing = preprocessing
        self.model = model
        self.index = index
        self.search = search


    def _precompute(self):
        """
        Precompute embeddings for all images and construct a K-D Tree.
        """
        all_embeddings = []
        all_metadata = []

        for image_path in glob(os.path.join(GALLERY_STORAGE, '*/*.jpg')):
            # Extract embeddings
            probe = Image.open(image_path)
            probe = preprocessing.process(probe)
            embedding = self.__predict(probe)
            all_embeddings.append(embedding)

            # Extract metadata
            base_name = os.path.basename(image_path)
            base_name = base_name.split('.')[0]
            image_id = base_name.split('_')[-1]
            name = base_name.split('_')[:-1]
            full_name = '_'.join(name)
            metadata = {'name': full_name, 'image_path': image_path}
            all_metadata.append(metadata)

            # Save embeddings
            model_name = os.path.splitext(os.path.basename(self.model.model_path))[0]
            embedding_filename = f"{full_name}_{image_id}_{model_name}.npy"
            save_path = os.path.join(EMBEDDINGS_STORAGE, model_name, embedding_filename)
            self.__save_embeddings(save_path, embedding)

    def __predict(self, probe):
        """Extract the embedding vector output from a preprocessed image."""
        embedding = self.model.extract(probe)
        return embedding

    def __save_embeddings(self, filename, embedding):
        """Store the embeddings in a numpy format with the given filename convention."""
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(filename, embedding)
        
    def _log_access(self, probe_image, results):
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "probe_image": probe_image,
            "results": results
        }
        log_filename = os.path.join(ACCESS_LOGS_STORAGE, f"log_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        with open(log_filename, 'w') as log_file:
            json.dump(log_entry, log_file)
            
    def search_gallery(self, probe, probe_filename, k=5):
        """
        Search for the nearest neighbors in the gallery.

        Args:
            probe (Image): The probe image to search for.
            k (int): The number of nearest neighbors to return.

        Returns:
            list: A list of tuples containing the individual's full name and the image filename.
        """
        probe_embedding = self.__predict(probe)
        neighbors = self.search.find_nearest_neighbors(probe_embedding, k)
        results = [(neighbor['name'], neighbor['image_path']) for _, neighbor in neighbors]
        self._log_access(probe_filename, results)
        return results

    def add_identity(self, image_path, name):
        try:
            probe = Image.open(image_path).convert('RGB')
            probe = self.preprocessing.process(probe)
            embedding = self.__predict(probe)
            base_name = os.path.basename(image_path)
            image_id = base_name.split('_')[-1].split('.')[0]
            full_name = '_'.join(base_name.split('_')[:-1])
            metadata = {'name': name, 'image_path': image_path}
            self.index.insert(embedding, metadata)
            model_name = os.path.splitext(os.path.basename(self.model.model_path))[0]
            embedding_filename = f"{full_name}_{image_id}_{model_name}.npy"
            save_path = os.path.join(EMBEDDINGS_STORAGE, model_name, embedding_filename)
            self.__save_embeddings(save_path, embedding)
        except Exception as e:
            return str(e)

    def remove_identity(self, name):
        image_pattern = os.path.join(GALLERY_STORAGE, f'{name}_*.jpg')
        for image_path in glob(image_pattern):
            os.remove(image_path)

        embedding_pattern = os.path.join(EMBEDDINGS_STORAGE, f'**/{name}_*.npy')
        for embedding_path in glob(embedding_pattern, recursive=True):
            os.remove(embedding_path)

        self._precompute()

    def get_access_logs(self, start_time, end_time):
        logs = []
        for log_file in glob(os.path.join(ACCESS_LOGS_STORAGE, '*.json')):
            with open(log_file, 'r') as f:
                log = json.load(f)
                log_time = datetime.strptime(log['timestamp'], "%Y-%m-%d %H:%M:%S")
                if start_time <= log_time <= end_time:
                    logs.append(log)
        return logs



if __name__ == "__main__":
    image_size = 224
    architecture = 'resnet_018'

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

    #probe_path = "simclr_resources/probe/Aaron_Sorkin/John_Smith_0002.jpg"
    #probe = Image.open(probe_path).convert('RGB')
    #probe = preprocessing.process(probe)
    #embedding = self.__predict(probe)

    #base_name = os.path.basename(probe_path)
    #base_name = base_name.split('.')[0]
    #image_id = base_name.split('_')[-1]
    #name = base_name.split('_')[:-1]
    #full_name = '_'.join(name)
    #embedding_filename = f"model_size_{image_size:03}_{architecture}/{full_name}/{full_name}_{image_id}.npy"
    #embedding_path = os.path.join(EMBEDDINGS_STORAGE, embedding_filename)

    #self._save_embeddings(embedding_path, embedding)
