import os
from glob import glob
import numpy as np
import cv2
import pandas as pd
import zipfile


class HardNegativeMiner:
    """
    A class to mine hard negative examples for a given model.

    Attributes:
        model: The model used for prediction.
        nms: Non-maximum suppression object.
        measure: Measure to evaluate predictions.
        dataset_dir: Directory containing the dataset.
        table: DataFrame to store the results.
    """

    def __init__(self, model, nms, measure, dataset_dir):
        """
        Initialize the HardNegativeMiner with model, nms, measure, and dataset directory.

        Args:
            model: The model used for prediction.
            nms: Non-maximum suppression object.
            measure: Measure to evaluate predictions.
            dataset_dir: Directory containing the dataset.
        """
        self.model = model
        self.nms = nms
        self.measure = measure
        self.dataset_dir = dataset_dir
        self.table = pd.DataFrame(columns=['annotation_file', 'image_file'] + self.measure.columns)

    def __read_annotations(self, file_path):
        """
        Read annotations from a text file.

        Args:
            file_path (str): Path to the annotation file.

        Returns:
            list: List of annotations in the format (class_label, x_center, y_center, width, height).
        """
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_label = int(parts[0])
                bbox = list(map(float, parts[1:]))
                annotations.append((class_label, *bbox))
        return annotations

    def __predict(self, image):
        """
        Make a prediction on the provided image using the model.

        Args:
            image (ndarray): The image to predict on.

        Returns:
            output: The model's prediction output.
        """
        output = self.model.predict(image)
        return output

    def __construct_table(self):
        """
        Construct a table with image files, annotation files, and measures.

        This method reads images and annotations, makes predictions,
        computes measures, and appends the results to the table.
        """
        for image_file, annotation_file in zip(
                sorted(glob(os.path.join(self.dataset_dir, "*.jpg"))),
                sorted(glob(os.path.join(self.dataset_dir, "*.txt")))):

            image = cv2.imread(image_file)
            annotation = self.__read_annotations(annotation_file)
            prediction = self.__predict(image)

            measures = self.measure.compute(prediction, annotation)

            self.table = self.table.append(
                {'annotation_file': annotation_file, 'image_file': image_file, **measures},
                ignore_index=True
            )

    def sample_hard_negatives(self, num_hard_negatives, criteria):
        """
        Sample hard negative examples based on the specified criteria.

        Args:
            num_hard_negatives (int): The number of hard negatives to sample.
            criteria (str): The criteria to sort and sample the hard negatives.

        Returns:
            DataFrame: A DataFrame containing the sampled hard negative examples.
        """
        self.__construct_table()
        self.table.sort_values(by=criteria, inplace=True, ascending=False)
        return self.table.head(num_hard_negatives)


    def augment_image(self, image):
        """
        Apply augmentations to an image.

        Args:
            image (ndarray): The image to augment.

        Returns:
            ndarray: The augmented image.
        """
        # Example augmentations: flip, rotate, change brightness, etc.
        augmented_images = []
        augmented_images.append(cv2.flip(image, 1))  # Horizontal flip
        augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))  # Rotate 90 degrees clockwise
        augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))  # Rotate 90 degrees counterclockwise
        
        # Change brightness
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 50)
        bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmented_images.append(bright_image)

        return augmented_images

    def augment_and_save(self, hard_negatives, save_dir):
        """
        Augment hard negative images and save them.

        Args:
            hard_negatives (DataFrame): DataFrame containing hard negative examples.
            save_dir (str): Directory to save the augmented images.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for idx, row in hard_negatives.iterrows():
            image = cv2.imread(row['image_file'])
            augmented_images = self.augment_image(image)

            for i, aug_img in enumerate(augmented_images):
                image_filename = os.path.basename(row['image_file']).replace('.jpg', f'_aug_{i}.jpg')
                save_path = os.path.join(save_dir, image_filename)
                cv2.imwrite(save_path, aug_img)

    def extract_zip(self, zip_path, extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if '__MACOSX' in member or member.endswith('.DS_Store'):
                    continue  # Skip unnecessary files and directories
            
                # Extract the file and handle long file names
                filename = os.path.basename(member)
                source = zip_ref.open(member)
                target = os.path.join(extract_to, filename[:255])  # Ensure file name length does not exceed 255 characters
            
                # Create directories if needed
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
            
                if not member.endswith('/'):  # It's a file
                    with open(target, "wb") as target_file:
                        with source:
                            target_file.write(source.read())
