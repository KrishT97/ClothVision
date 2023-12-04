import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image


# En el __init__ la clase recibe donde esta el class_dict.csv, donde esta el encoder de onehot (joblib) y una o
# múltiples imágenes (img* como argumento, el asterisco implica 1 o más en python) y las devuelve en el mismo orden
# en dos listas, la primera tiene la imágen con sus colores de segmentacion en lugar de codificación one hot
# y la segunda contiene que clases tenía cada imagen en una lista de tuplas [(gorro pantalon chaqueta), (abrigo, vaquero, gafas)]
# hacer tan óptima como sea posible.


class LabelDecoder:

    # Se crea el class_dict y el encoder de onehot pasado a array
    def __init__(self, class_dict_path, encoder_path):
        self.class_dict = pd.read_csv(class_dict_path)
        self.onehot_to_array = self.__get_onehot_to_array(encoder_path)

    # Paso el onehot de las clases a array, esto me sirve para decodificar las imagenes
    def __get_onehot_to_array(self, encoder_path):
        encoder = joblib.load(encoder_path)
        encoded_df = self.__get_class_dict_dataframe(encoder)
        encoded_df["byte_value"] = encoded_df["array_value"].apply(lambda x: x.tobytes())
        return encoded_df[["byte_value", "One-hot"]].set_index('byte_value').to_dict()["One-hot"]

    def __get_class_dict_dataframe(self, encoder):
        codes = encoder.transform(self.class_dict["class_name"].values.reshape(-1, 1))

        encoded_df = pd.concat([self.class_dict, pd.Series(list(codes[:]))], axis=1).rename(
            columns={0: "One-hot"},
            inplace=False)
        encoded_df["array_value"] = encoded_df[["r", "g", "b"]].apply(
            lambda x: np.array([x.r, x.g, x.b], dtype=np.int32), axis=1)
        encoded_df["tensor_value"] = encoded_df[["r", "g", "b"]].apply(
            lambda x: torch.tensor([x.r, x.g, x.b], dtype=torch.float32), axis=1)
        return encoded_df

    def decode_labels(self, *images):
        decoded_images = [self.__decode_image(image) for image in images]
        class_lists = [self.__get_classes(image) for image in decoded_images]
        return decoded_images, class_lists

    def __decode_image(self, image):
        return torch.stack([
            torch.stack([self.onehot_to_array[pixel.tobytes()] for pixel in row]) for row in image
        ])

    def __get_classes(self, image):
        return tuple(
            self.class_dict.loc[(self.class_dict[['r', 'g', 'b']] == image.numpy()).all(axis=1), 'class_name'].values)



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