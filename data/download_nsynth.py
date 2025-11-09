import requests
import tarfile
from pathlib import Path
import time
from tqdm import tqdm

NSYNTH_TEST_URL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
TARGET_DIR = Path(__file__).resolve().parent / "nsynth"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, target_path):
    print(f"Downloading {url} → {target_path}")
    response = requests.get(url, stream=True)
    total_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MB

    with open(target_path, "wb") as f, tqdm(
        total=total_bytes,
        unit='B',
        unit_scale=True,
        desc=target_path.name
    ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))
    print("Download complete.")


def extract_tar_gz(path, extract_to):
    print(f"Extracting {path} → {extract_to}")
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting", unit="file"):
            tar.extract(member, path=extract_to)
    print("Extraction complete.")
    path.unlink()  # remove archive after extraction

if __name__ == "__main__":
    start = time.time()
    archive_path = TARGET_DIR / "nsynth-test.jsonwav.tar.gz"
    extracted_dir = TARGET_DIR / "nsynth-test"

    if extracted_dir.exists():
        print(f"nsynth-test already exists at {extracted_dir}, skipping download.")
    else:
        download_file(NSYNTH_TEST_URL, archive_path)
        extract_tar_gz(archive_path, TARGET_DIR)

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f} min)")