# fasttfidf

fasttfidf is a Python library that provides TF-IDF vectorization with automatic memory management and SIMD acceleration. It processes datasets larger than RAM using memory-mapped files and streaming architecture, making it practical to work with multi-gigabyte text corpora on commodity hardware

## Key Features

- **Memory-efficient processing**: Handles datasets larger than available RAM through streaming and automatic memory management
- **SIMD optimization**: Leverages AVX2 (x86_64) and NEON (ARM) instruction sets for accelerated text processing
- **Multiprocessing**: Parallel vocabulary building across CPU cores with automatic load balancing
- **Batch training support**: Train models incrementally without loading full dataset into memory
- **Vocabulary exploration**: Built-in methods for analyzing and querying learned vocabularies

## Limitations

- **CSV and Parquet only**: Requires CSV or Parquet files with a `text` column header
- **No preprocessing**: Does not perform stopword removal, stemming, or lemmatization - input text must be preprocessed
- **Batch processing required**: Transform returns raw components (data, indices, indptr) that require manual conversion to sparse matrices
- **Manual IDF application**: IDF weighting and normalization must be applied manually during transformation (example given)
- **File-based only**: Cannot process in-memory data structures or Python dataframes/lists directly

## Installation

### Step 1: Library Installation

Install Apache Arrow C++ library (Version 22):

**macOS (Homebrew):**
```bash
brew install apache-arrow@22
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y -V ca-certificates lsb-release wget
wget https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-$(lsb_release -cs).deb
sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release -cs).deb
sudo apt update
sudo apt install -y libarrow-dev libparquet-dev pkg-config
```

### Step 2: Python ENV Setup Using Conda (Recommended)

```bash
conda create -n fasttfidf python=3.11 -y
conda activate fasttfidf
conda install -c conda-forge arrow-cpp pyarrow psutil pybind11 pytest -y
git clone https://github.com/purijs/fasttfidf
cd fasttfidf
/Users/jaskaransinghpuri/anaconda3/envs/fasttfidf/bin/pip install --no-build-isolation -e .

Once installed, make sure your Python Kernel points to: /Users/$USER/anaconda3/envs/fasttfidf/bin/python3.11
```

**OR**

### Step 2: Python ENV Setup Using Poetry

Run these:
```bash
export ARROW_HOME=$(brew --prefix apache-arrow@22)
export PKG_CONFIG_PATH="${ARROW_HOME}/lib/pkgconfig:$PKG_CONFIG_PATH"
```

pyproject.toml
```
[tool.poetry]
name = "fasttfidf"
version = "0.1.0"
description = "High-performance TF-IDF for large-scale text datasets"
authors = ["Jaskaran Singh Puri"]
license = "MIT"
readme = "README.md"
packages = []  # C++ extension, no pure Python packages

[tool.poetry.dependencies]
python = ">=3.9,<3.13"
numpy = ">=1.19.0"
scipy = ">=1.5.0"
pyarrow = ">=22.0.0,<23.0.0"
psutil = ">=5.9.0"
pybind11 = ">=2.10.0"

[tool.poetry.group.dev.dependencies]
pytest = ">=7.0.0"

[build-system]
requires = [
    "setuptools>=68",
    "pybind11>=2.10.0"
]
build-backend = "setuptools.backends.legacy:build"
```

Then Run:
```bash
poetry install
poetry run pip install --no-build-isolation -e .
```

Requirements:
- Python 3.9+
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- C++17 compatible compiler
- Apache Arrow C++ library (version: 22)

## Quick Start

### Step 1: Fit and Save Model

```python
import fasttfidf_csv

# Fit the vectorizer on your training data
vec = fasttfidf_csv.TfidfVectorizer()
vec.fit('train.csv', num_processes=0) # use all cores

# Save model for later use
vec.save('model.tfidf')
```

### Step 2: Transform Data

fasttfidf provides two transformation workflows depending on your use case.

#### Option A: Batch Training (Memory-Efficient)

For datasets larger than RAM, train models incrementally:

```python
import fasttfidf_csv
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
import numpy as np

# Load model
vec = fasttfidf_csv.TfidfVectorizer()
vec.load('model.tfidf')

# Get IDF weights once
idf_weights = vec.get_idf_array().astype(np.float32)
n_features = len(idf_weights)

# Initialize incremental learner
model = SGDClassifier(loss='log_loss', max_iter=1)

# Stream and train in batches
vec.open_stream('train.csv')
batch_size = 128 * 1024 * 1024  # 128 MB

while True:
    # Get raw term frequencies
    batch = vec.get_batch(batch_size)
    if batch is None:
        break
    
    data, indices, indptr = batch
    n_docs = len(indptr) - 1
    
    # Build sparse matrix
    X = csr_matrix((data, indices, indptr), 
                   shape=(n_docs, n_features))
    
    # Apply IDF weighting and L2 normalization
    X = X.astype(np.float32)
    X.data *= idf_weights[X.indices]
    normalize(X, norm='l2', copy=False)
    
    # Incremental training
    model.partial_fit(X, y_batch, classes=np.unique(y_train))
```

#### Option B: Full Matrix (For Smaller Datasets)

When the full TF-IDF matrix fits in memory:

```python
from scipy.sparse import vstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import numpy as np

# Load model
vec = fasttfidf_csv.TfidfVectorizer()
vec.load('model.tfidf')

# Get IDF weights
idf_weights = vec.get_idf_array().astype(np.float32)
n_features = len(idf_weights)

# Collect all batches
matrices = []
vec.open_stream('train.csv')
batch_size = 500 * 1024 * 1024  # 500 MB

while True:
    batch = vec.get_batch(batch_size)
    if batch is None:
        break
    
    data, indices, indptr = batch
    n_docs = len(indptr) - 1
    
    # Build sparse matrix
    X = csr_matrix((data, indices, indptr), 
                   shape=(n_docs, n_features))
    
    # Apply IDF and normalize
    X = X.astype(np.float32)
    X.data *= idf_weights[X.indices]
    normalize(X, norm='l2', copy=False)
    
    matrices.append(X)

# Combine all batches
X_train = vstack(matrices)

# Train model on full matrix
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Vocabulary Exploration

```python
# Get vocabulary statistics
stats = vec.get_vocab_stats()
print(f"Vocabulary size: {stats['vocab_size']}")
print(f"Total documents: {stats['total_docs']}")

# Find rarest terms
rare_words = vec.get_top_idf_words(n=10)
for word, idf in rare_words:
    print(f"{word}: {idf:.3f}")

# Find most common terms
common_words = vec.get_bottom_idf_words(n=10)

# Search vocabulary
results = vec.search_words('machine', max_results=50)

# Check specific terms
idf_value = vec.get_word_idf('computer')
doc_freq = vec.get_word_df('computer')
```

## Input Format Requirements

### CSV Format

fasttfidf expects CSV files with a header row and a `text` column:

```csv
text
This is the first document.
This document is the second document.
And this is the third one.
```

### Parquet Format

fasttfidf also supports Apache Parquet files with a `text` column. Use the `fasttfidf_parquet` module for parquet files:

```python
import fasttfidf_parquet

vec = fasttfidf_parquet.TfidfVectorizer()
vec.fit('/path/to/parquet/files/', num_processes=0) # or file.parquet
vec.save('model.tfidf')

# Transform works the same way
vec.open_stream('test.parquet')
batch = vec.get_batch(128 * 1024 * 1024)
```

**Important**: The library does not perform any text preprocessing. Your files must contain pre-processed text with stopwords removed, text lowercased, and any other desired preprocessing already applied.

## API Reference

### TfidfVectorizer

#### Training Methods

- `fit(filename, num_processes=0, min_df=1, max_df=0, max_features=0, verbose=True)` - Build vocabulary from CSV file
  - `filename`: Path to CSV file
  - `num_processes`: Number of workers (0 = auto-detect)
  - `min_df`: Minimum document frequency
  - `max_df`: Maximum document frequency (0 = no limit)
  - `max_features`: Limit vocabulary size (0 = no limit)
  - `verbose`: Print progress messages

- `save(filename)` - Save model to disk as text file
- `load(filename)` - Load model from disk

#### Transform Methods

- `open_stream(filename)` - Open CSV file for streaming transformation
- `get_batch(batch_size_bytes)` - Get next batch of raw term frequencies
  - Returns: `(data, indices, indptr)` tuple of NumPy arrays, or `None` when stream ends
  - `data`: uint16 array of term frequencies
  - `indices`: int32 array of column indices
  - `indptr`: int32 array of row pointers (CSR format)

#### Vocabulary Methods

- `get_vocabulary()` - Return vocabulary as dict mapping word -> index
- `get_idf()` - Return IDF values as Python list
- `get_idf_array()` - Return IDF values as NumPy array
- `get_feature_names()` - Return feature names in index order
- `get_vocab_size()` - Return vocabulary size
- `get_total_docs()` - Return total documents processed during fit

#### Exploration Methods

- `get_vocab_stats()` - Get vocabulary statistics dict
- `get_top_idf_words(n=10)` - Get n words with highest IDF (rarest)
- `get_bottom_idf_words(n=10)` - Get n words with lowest IDF (most common)
- `get_word_idf(word)` - Get IDF value for specific word
- `get_word_df(word)` - Get document frequency for word
- `search_words(pattern, max_results=100)` - Search vocabulary by substring
- `get_words_in_idf_range(min_idf, max_idf, max_results=1000)` - Filter by IDF range
- `get_words_in_df_range(min_df, max_df, max_results=1000)` - Filter by document frequency
- `has_word(word)` - Check if word exists in vocabulary
- `get_random_words(n=10, seed=42)` - Get random vocabulary sample
- `export_vocabulary_with_idf()` - Export vocabulary as dict with IDF values

## Architecture

fasttfidf uses a three-stage pipeline optimized for large-scale processing:

1. **Vocabulary Building**: Memory-mapped file access with multiprocessing and dynamic sub-batching prevents out-of-memory errors on large datasets. Worker processes use adaptive memory management to stay within available RAM.

2. **IDF Calculation**: Inverse document frequency values are computed once during fit and cached in the model file for efficient transformation.

3. **Streaming Transformation**: Zero-copy batch processing returns CSR sparse matrix components (data, indices, indptr) that can be incrementally processed or combined.

## Testing

Run the test suite:

```bash
pytest tests.py -v
```

## License

This project is distributed under the MIT License. See LICENSE file for details.

## Citation

If you use fasttfidf in a scientific publication, please cite:

```bibtex
@software{fasttfidf2025,
  author = {Puri, Jaskaran Singh},
  title = {fasttfidf: High-performance TF-IDF for large-scale text datasets},
  year = {2025},
  url = {https://github.com/purijs/fasttfidf}
}
```
