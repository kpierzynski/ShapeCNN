from torch.utils.data import Dataset
from torch import from_numpy, tensor
from PIL import Image
from pathlib import Path
from numpy import array, asarray


class ShapeDataset(Dataset):
    def __init__(self, path, transform=None):
        if not Path(path).is_dir():
            raise ValueError(f"Invalid path: {path}")

        self.path = path
        self.transform = transform
        self.label_mapping = {"circle": 0, "square": 1, "triangle": 2}

        self.classes = "circle", "square", "triangle"

        for class_name in self.classes:
            dir_name = class_name + "s"
            if not Path(path, dir_name).is_dir():
                raise ValueError(f"Missing class data directory: {path} {class_name}")

        self.data = []
        for class_name in self.classes:
            dir_name = class_name + "s"
            for file in Path(path, dir_name).iterdir():
                if file.is_file():
                    image = Image.open(file).convert("RGB")
                    self.data.append({"data": image, "label": class_name})

    def index2label(self, index):
        return next(key for key, value in self.label_mapping.items() if value == index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        data = item["data"]
        label = item["label"]

        if self.transform:
            data = self.transform(data)

        return data, self.label_mapping[label]


if __name__ == "__main__":
    dataset = ShapeDataset("./dataset")

    test_element = dataset[0]
    data, label = test_element

    print(data)
    print(label)
