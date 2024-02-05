from pathlib import Path

from LabelManagement.LabelEncoder import LabelEncoder


def preprocess_data():
    Path("./data/labels/processed_pixel_labels").mkdir(parents=True, exist_ok=True)
    LabelEncoder("./data/labels/pixel_level_labels_colored", "./data/labels/processed_pixel_labels/",
                 "./data/class_dict.csv", "./one_hot_encoder.joblib").preprocess()
