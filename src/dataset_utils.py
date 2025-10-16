import kaggle
import keras
from pathlib import Path

RAF_DB_DATASET = "shuvoalok/raf-db-dataset"
FER_2013_DATASET = "msambare/fer2013"
DATA_ROOT = Path("../data")

def _download_dataset(dataset_url: str) -> Path:
    dataset_name = dataset_url.split("/")[1]
    dataset_path = DATA_ROOT / dataset_name
    DATA_ROOT.mkdir(exist_ok=True)
    if dataset_path.exists():
        print("Dataset is already downloaded")
        return dataset_path
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=dataset_url, path=dataset_path, unzip=True)
        print(f"Downloaded dataset to {dataset_path}")
        return dataset_path
    except Exception as e:
        print("Failed to download dataset: ", e)
        raise

def get_raf_db_dataset(image_size=(128,128), batch_size=32):
    raf_db_path = _download_dataset(RAF_DB_DATASET)
    test_path = raf_db_path/"DATASET/test"
    train_path = raf_db_path/"DATASET/train"

    train_ds = keras.utils.image_dataset_from_directory(
        train_path,
        image_size=image_size,
        batch_size=batch_size
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_path,
        image_size=image_size,
        batch_size=batch_size
    )
    return train_ds, test_ds

def get_fer_2013_dataset(image_size=(128,128), batch_size=32):
    fer_2013_path = _download_dataset(FER_2013_DATASET)
    test_path = fer_2013_path/"test"
    train_path = fer_2013_path/"train"

    train_ds = keras.utils.image_dataset_from_directory(
        train_path,
        image_size=image_size,
        batch_size=batch_size
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_path,
        image_size=image_size,
        batch_size=batch_size
    )
    return train_ds, test_ds

# if __name__ == "__main__":
#     get_fer_2013_dataset()