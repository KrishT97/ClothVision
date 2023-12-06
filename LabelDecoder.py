import joblib
import numpy as np
import pandas as pd
import torch


class LabelDecoder:

    # Se crea el class_dict y el encoder de onehot pasado a array
    def __init__(self, class_dict_path, encoder_path):
        self.encoder = joblib.load(encoder_path)
        self.class_dict = self.__get_class_dict_dataframe(class_dict_path)
        self.onehot_to_array = self.__get_onehot_to_array()

    def __get_onehot_to_array(self, ):
        array_to_onehot = self.class_dict[["byte_value", "One-hot"]].set_index('byte_value').to_dict()["One-hot"]
        return {v.tobytes(): k for k, v in array_to_onehot.items()}

    def __get_class_dict_dataframe(self, class_dict_path):
        class_dict = pd.read_csv(class_dict_path)
        codes = self.encoder.transform(class_dict["class_name"].values.reshape(-1, 1)).astype(np.uint8)
        class_dict = pd.concat([class_dict, pd.Series(list(codes[:]))], axis=1).rename(
            columns={0: "One-hot"},
            inplace=False)
        class_dict["array_value"] = class_dict[["r", "g", "b"]].apply(
            lambda x: np.array([x.r, x.g, x.b], dtype=np.uint8), axis=1)
        class_dict["byte_value"] = class_dict['array_value'].apply(lambda array: array.tobytes())
        return class_dict

    def decode_labels(self, *images):
        result = []

        for image in images:
            decoded_image = self.__decode_image(image)
            classes = self.__get_classes(decoded_image)
            result.append((decoded_image, classes))
        return result

    def __decode_image(self, image):
        return np.apply_along_axis(lambda pixel: np.frombuffer(self.onehot_to_array[pixel.tobytes()], dtype=np.uint8),
                                   0, np.array(image.cpu(), np.uint8)).transpose(1, 2, 0)

    def __get_classes(self, decoded):
        image_df = pd.DataFrame({'array_value': decoded.reshape(-1, 3).tolist()}).drop_duplicates()
        image_df['byte_value'] = image_df['array_value'].apply(lambda x: np.array(x, dtype=np.uint8).tobytes())
        return pd.merge(image_df, self.class_dict, how='inner', left_on=['byte_value'], right_on=['byte_value'])[
            "class_name"].tolist()




"""
from LabelDecoder import LabelDecoder
# Uso del LabelDecoder
decoder = LabelDecoder(class_dict_path=DATASET_DIR + "/class_dict.csv", encoder_path='one_hot_encoder.joblib')
image1 = CustomClothingDataset(root_dir=DATASET_DIR).__getitem__(0)[1]
image2 = CustomClothingDataset(root_dir=DATASET_DIR).__getitem__(0)[2]
image3 = CustomClothingDataset(root_dir=DATASET_DIR).__getitem__(0)[3]
# Decodificar las imágenes
decoded_images, class_lists = decoder.decode_labels(image1, image2, image3)
for decoded_image, classes in zip(decoded_images, class_lists):
    print(f"Decoded Image: {decoded_image}")
    print(f"Classes: {classes}")
"""
