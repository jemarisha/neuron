import os

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch

from utils.enums import PtDatasetTypes


class SimpleDatasetLoader(Dataset):
    def __init__(self,
                 x_csv_file: str,
                 y_csv_file: str,
                 device_str: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        DataLoader для работы с простыми характеристиками из csv файлов
        :param x_csv_file: путь до csv файла с x характеристиками
        :param y_csv_file: путь до csv файла с y характеристиками
        :param device_str: строковое название устройства, на которое будет загружен dataset (cuda/cpu)
        """
        self.device = torch.device(device_str)
        self.data_x = torch.tensor(pd.read_csv(x_csv_file).to_numpy(),
                                   dtype=torch.float32,
                                   device=self.device)

        self.data_y = torch.tensor(pd.read_csv(y_csv_file)[["x1", "x2"]].to_numpy(),
                                   dtype=torch.float32,
                                   device=self.device)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)


class PtDatasetLoader(Dataset):
    def __init__(self,
                 is_train: bool,
                 dataset_type: PtDatasetTypes,
                 path_to_dataset: str = '/cephfs/projects/share/dataset/dataset_pt_final',
                 device_str: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        DataLoader для работы с графовым представлением логических схем из .pt файлов
        :param is_train: True если создаём выборку для обучения и False, если для тестирования
        :param dataset_type: как будет разделён OpenABC-D датасет (см. Подробнее комментарии в PtDatasetTypes)
        :param path_to_dataset: путь до **папки** с .pt файлами (.pt файлы для каждой модель обязательно должны быть
        внутри своей поддиректории)
        :param device_str: строковое название устройства, на которое будет загружен dataset (cuda/cpu)
        """
        self.path_to_dataset = path_to_dataset
        self.dataset_dirs = os.listdir(self.path_to_dataset)
        self.device = torch.device(device_str)
        self.pt_paths = []


        # Формируем список всех pt файлов, включая относительный путь в название pt файла
        for dataset_dir in self.dataset_dirs:
            dataset_dir_path = f'{self.path_to_dataset}/{dataset_dir}'
            pt_list_one_model = [f'{dataset_dir_path}/{pt_file_name}' for pt_file_name in os.listdir(dataset_dir_path)]
            self.expl_in_model = len(pt_list_one_model)
            self.pt_paths.extend(pt_list_one_model)

        # Распределяем либо 1000 на 500, либо 19 на 10 моделей для обучения и тестов
        match dataset_type:
            case PtDatasetTypes.partly:
                part = 2 / 3 if is_train else 1 / 3
                sep = round(self.expl_in_model * part)
                self.pt_paths = [self.pt_paths[i: i + sep] for i in range(0, len(self.pt_paths), self.expl_in_model)]
            case PtDatasetTypes.test:
                self.pt_paths = self.pt_paths[:50]
            case PtDatasetTypes.sfl:
                sep = round(len(self.pt_paths) * 2 / 3)
                self.pt_paths = self.pt_paths[:sep] if is_train else self.pt_paths[sep:]

        # Приводим массив к линейной структуре
        self.pt_paths = list(np.array(self.pt_paths).flatten())

    def __getitem__(self, index):
        return torch.load(self.pt_paths[index], map_location=self.device)

    def __len__(self):
        return len(self.pt_paths)
