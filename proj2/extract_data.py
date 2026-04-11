import json
import os
import zipfile
from config import DATA_DIR

def extract(zip_name):
    filepath = DATA_DIR / zip_name
    with zipfile.ZipFile(filepath) as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(filepath)


def _create_filename_to_img_path(train: bool = True):
    if train:
        json_path = DATA_DIR / "train" / "train_mapping.json"
        img_dir = DATA_DIR / "train" / "images"
    else:
        json_path = DATA_DIR / "test" / "test_mapping.json"
        img_dir = DATA_DIR / "test" / "images"

    if json_path.exists():
        print(f"Mapping {json_path.name} already exists")
        return

    name_mapping = {}
    valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']

    summator = 0

    for file in img_dir.iterdir():

        if not file.is_file():
            continue

        if file.suffix not in valid_extensions:
            continue

        name_mapping[file.stem] = file.name
        summator += 1

    print(f"Found and mapped {summator} images")

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(name_mapping, json_file, indent=4, ensure_ascii=False)

    print(f"Successfully mapped to file {json_path.name}")


def get_mappings(train: bool = True):
    if train:
        json_path = DATA_DIR / "train" / "train_mapping.json"
    else:
        json_path = DATA_DIR / "test" / "test_mapping.json"

    with open(json_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)

if __name__ == "__main__":
    # extract(zip_name="D-Fire.zip")
    _create_filename_to_img_path(True)
    _create_filename_to_img_path(False)