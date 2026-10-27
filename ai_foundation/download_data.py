import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path("data")
FILE_PATH = DATA_DIR / "shakespeare.txt"


def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"Already downloaded: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")


def describe(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    words = text.split()
    print(f"Characters:        {len(text):,}")
    print(f"Unique characters: {len(set(text))}")
    print(f"Words:             {len(words):,}")
    print(f"Unique words:      {len(set(words)):,}")
    print("\nFirst 300 characters:\n" + "-" * 40)
    print(text[:300])


if __name__ == "__main__":
    download(URL, FILE_PATH)
    describe(FILE_PATH)