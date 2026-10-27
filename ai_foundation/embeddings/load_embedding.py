"""A utility function to load Gemma embeddings from a URL."""

import io
from typing import List, Tuple
import urllib.request
import numpy as np


def load_gemma_embeddings(url: str) -> Tuple[np.ndarray, List[str]]:
  """Load Gemma embeddings from a URL.

  Args:
    url: The URL to load the embeddings from.

  Returns:
    A tuple containing the embeddings and labels.
  """
  with urllib.request.urlopen(url) as response:
    buffer = io.BytesIO(response.read())
    bundle = np.load(buffer)
  return bundle['embeddings'], bundle['labels'].tolist()