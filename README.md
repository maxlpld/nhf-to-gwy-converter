# NHF to GWY Converter

Small command-line tool to convert NanoSurf `.nhf` HDF5 files into Gwyddion `.gwy` files and generate compact overview PDFs.

The script is designed for AFM data exported as `.nhf` files. It searches for image-like numeric datasets, keeps forward/trace and backward/retrace channels separate, writes the detected channels to a `.gwy` file, and creates one `nhf_overview.pdf` per folder containing `.nhf` files.

## Features

- Converts one `.nhf` file into one `.gwy` file.
- Recursively processes all `.nhf` files when a folder is provided.
- Stores all detected image channels in the output `.gwy` file.
- Preserves forward/backward scan directions when they can be inferred from metadata or HDF5 paths.
- Creates overview PDFs with grouped forward/backward panels.
- Uses common color scales, colorbars, scale bars, and square image panels in the PDF overview.

## Requirements

- Python 3.9 or newer recommended
- `h5py`
- `numpy`
- `matplotlib`
- `gwyfile`

## Installation

From the folder where you want to keep the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install h5py numpy matplotlib gwyfile
```

## Usage

Run the script directly:

```bash
python3 convert_nhf_to_gwy.py
```

The script will show the current default path and ask you to press Enter to use it or paste another `.nhf` file/folder path.

You can also pass the input path directly:

```bash
python3 convert_nhf_to_gwy.py /path/to/file_or_folder
```

Examples:

```bash
python3 convert_nhf_to_gwy.py measurement.nhf
python3 convert_nhf_to_gwy.py /home/user/AFM_measurements/
```

## Output

For each input `.nhf` file, the script creates:

```text
measurement.gwy
```

next to the original `.nhf` file.

For each folder containing processed `.nhf` files, it also creates:

```text
nhf_overview.pdf
```

The PDF contains overview pages with the detected logical channels. Forward and backward images are shown side by side when matching channels are found.

## Configuration

Some useful options can be edited near the top of `convert_nhf_to_gwy.py`:

```python
DEFAULT_INPUT_PATH = Path("/path/to/default/folder")
OVERWRITE_EXISTING = True
CHANNEL_GROUPS_PER_PDF_PAGE = 4
```

- `DEFAULT_INPUT_PATH`: default file or folder shown when the script starts.
- `OVERWRITE_EXISTING`: if `True`, existing `.gwy` and `nhf_overview.pdf` files are overwritten.
- `CHANNEL_GROUPS_PER_PDF_PAGE`: number of logical channel groups shown per PDF page.

## Suggested project structure

```text
nhf-to-gwy-converter/
├── convert_nhf_to_gwy.py
├── README.md
└── .gitignore
```

A minimal `.gitignore` could contain:

```gitignore
.venv/
__pycache__/
*.pyc
*.gwy
*.pdf
*.nhf
```

Remove `*.nhf`, `*.gwy`, or `*.pdf` from `.gitignore` only if you really want to store raw data or generated outputs in the repository.

## Notes

- The script expects NanoSurf `.nhf` files to be HDF5-based.
- The conversion depends on metadata available inside the file. If metadata are incomplete, some units or scan sizes may fall back to default values.
- Large experimental data files are usually better kept outside the GitHub repository.
