import os

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image


class LabelEncoder:

    def __init__(self, labels_path, labels_new_path, class_dict_path, encoder_path):
        self.labels_path = labels_path
        self.encoder = joblib.load(encoder_path)
        self.labels_new_path = labels_new_path
        self.onehot_encoder_dict = self.__get_encoder_from_class_dict_dataframe(
            self.__get_class_dict_dataframe(class_dict_path))
        self.label_files = os.listdir(labels_path)

    def preprocess(self):
        for label in self.label_files:
            labels_path = os.path.join(self.labels_path, label)
            self.save_label(self.preprocess_label(Image.open(labels_path)), labels_path)

    def get_preprocessed(self):
        preprocessed = []
        for label in self.label_files:
            labels_path = os.path.join(self.labels_path, label)
            preprocessed.append(self.preprocess_label(Image.open(labels_path)))
        return preprocessed

    def preprocess_label(self, label):
        label = np.asarray(label, dtype=np.uint8)
        codified = np.apply_along_axis(lambda pixel: self.onehot_encoder_dict[pixel.tobytes()], 2, label)
        return torch.tensor(codified, dtype=torch.uint8)

    def save_label(self, codified_label, labels_path):
        new_file_name = os.path.basename(labels_path)[:-4] + ".pt"
        torch.save(codified_label, os.path.join(self.labels_new_path, new_file_name))

    def __get_class_dict_dataframe(self, class_dict_path):
        dataframe = pd.read_csv(class_dict_path)
        codes = self.encoder.fit_transform(dataframe["class_name"].values.reshape(-1, 1))

        encoded_df = pd.concat([dataframe, pd.Series(list(codes[:]))], axis=1).rename(columns={0: "One-hot"},
                                                                                      inplace=False)
        encoded_df["array_value"] = encoded_df[["r", "g", "b"]].apply(lambda x: np.array([x.r, x.g, x.b], dtype=np.int32), axis=1)
        encoded_df["tensor_value"] = encoded_df[["r", "g", "b"]].apply(
            lambda x: torch.tensor([x.r, x.g, x.b], dtype=torch.float32), axis=1)
        return encoded_df

    def __get_encoder_from_class_dict_dataframe(self, encoded_df):
        encoded_df["byte_value"] = encoded_df["array_value"].apply(lambda x: x.tobytes())
        return encoded_df[["byte_value", "One-hot"]].set_index('byte_value').to_dict()["One-hot"]


#LabelEncoder("./data/labels/pixel_level_labels_colored", "./data/labels/processed_pixel_labels/",
#                  "./data/class_dict.csv", "./one_hot_encoder.joblib").preprocess()
