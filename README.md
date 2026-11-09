# NLP & Computer Vision — CDAC PGCP-AI (Aug 2026)

Course materials, lecture notes, and assignments for the **Natural Language Processing & Computer Vision** module.

## 📁 Repository Structure

```
NLP_CDAC/
├── notes/                          # Ma'am's lecture content (organized by day)
│   ├── day-01/
│   │   ├── README.md               # Day summary & topic index
│   │   ├── slides/                 # .pptx lecture slides
│   │   ├── code/                   # .py lab exercises
│   │   └── pdfs/                   # PDF guides / output references
│   ├── day-02/
│   │   └── ...
│   └── day-NN/
│
├── assignments/                    # Your assignment submissions
│   ├── assignment-01/
│   │   ├── Day_1_Assignment.docx   # Original question paper
│   │   └── assignment_01.ipynb     # Your solution notebook
│   ├── assignment-02/
│   │   └── ...
│   └── assignment-NN/
│
├── reference-books/                # Textbooks & reference PDFs (gitignored)
├── raw-downloads/                  # Original zips from CloudFoundry (gitignored)
├── scripts/                        # Helper scripts
│   ├── add-day.ps1                 # Organize a new day's zip
│   └── new-assignment.ps1          # Create a new assignment notebook
├── requirements.txt                # Python dependencies
└── .gitignore
```

## 🚀 Quick Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 📥 Adding a New Day's Content

When you download a new zip from CloudFoundry:

```powershell
# Example: Downloaded "DAY 2.zip" to Downloads
.\scripts\add-day.ps1 -ZipPath "C:\Users\conne\Downloads\DAY 2.zip" -DayNumber 2
```

This will:
1. Extract the zip
2. Sort `.py` files → `notes/day-02/code/`
3. Sort `.pdf` files → `notes/day-02/pdfs/`
4. Sort `.pptx` files → `notes/day-02/slides/`
5. Archive the zip → `raw-downloads/`

## 📝 Adding a New Assignment

```powershell
.\scripts\new-assignment.ps1 -Number 2 -Title "Text Classification with NLTK"
```

## 📚 CloudFoundry Source

Content downloaded from:
```
https://cf.cdacb.in/index.php/s/l0rOWydcrtrzkiI?path=%2F002_Modulewise%20Contents%2F006_Natural%20Language%20Processing%20%26%20Computer%20Vision
Password: AI@EC2
```

## 📖 Reference Books
- Bhattacharyya & Joshi — *Natural Language Processing* (Wiley India, 2023)
- Pushpak Bhattacharyya — *Natural Language Processing*
- Shlomo Vaknin — *The Big Book of NLP*
- CDAC Syllabus PDF

