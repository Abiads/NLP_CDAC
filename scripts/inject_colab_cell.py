import json
import os

notebooks = [
    (
        r'assignments/assignment-01/assignment_01.ipynb',
        '/content/drive/MyDrive/NLP_CDAC/assignments/assignment-01',
        ['deep-translator', 'nltk']
    ),
    (
        r'assignments/assignment-02/Session3_Lab_Disneyland_Reviews.ipynb',
        '/content/drive/MyDrive/NLP_CDAC/assignments/assignment-02',
        ['pandas', 'nltk', 'spacy', 'matplotlib']
    ),
    (
        r'notes/day-01/day_01_lab_walkthrough.ipynb',
        '/content/drive/MyDrive/NLP_CDAC/notes/day-01',
        ['spacy', 'nltk', 'matplotlib', 'textblob']
    ),
    (
        r'notes/day-02/Session3_Lab_Disneyland_Reviews.ipynb',
        '/content/drive/MyDrive/NLP_CDAC/notes/day-02',
        ['pandas', 'nltk', 'spacy', 'matplotlib']
    ),
]

for nb_rel, drive_subdir, pkgs in notebooks:
    nb_path = os.path.join(r'c:\Users\conne\Downloads\NLP_CDAC', nb_rel)
    if not os.path.exists(nb_path):
        continue

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    pkg_str = ' '.join(pkgs)
    source_lines = [
        "# [Google Colab & Drive Setup - cdac74abhlak@gmail.com]\n",
        "# Run this cell if executing on Google Colab to mount Google Drive\n",
        "import os\n",
        "try:\n",
        "    import google.colab\n",
        "    IN_COLAB = True\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        f"    drive_dir = '{drive_subdir}'\n",
        "    os.makedirs(drive_dir, exist_ok=True)\n",
        "    os.chdir(drive_dir)\n",
        "    print('[OK] Google Drive linked for cdac74abhlak@gmail.com')\n",
        "    print('[DIR] Working Directory:', os.getcwd())\n",
        f"    # Uncomment below to install required packages on Colab runtime:\n",
        f"    # !pip install -q {pkg_str}\n",
        "except ImportError:\n",
        "    IN_COLAB = False\n",
        "    print('[OK] Running locally in VS Code / Antigravity IDE (Account: cdac74abhlak@gmail.com)')\n"
    ]

    colab_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines
    }

    # Only insert if not already present
    first_cell_src = "".join(nb['cells'][1].get('source', [])) if len(nb['cells']) > 1 else ""
    if 'Google Colab & Drive Setup' not in first_cell_src:
        nb['cells'].insert(1, colab_cell)
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Added Colab setup cell to: {nb_rel}")
    else:
        print(f"Colab setup cell already present in: {nb_rel}")
