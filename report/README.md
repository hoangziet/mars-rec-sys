# FPT University LaTeX Thesis Template

A professional, well-organized LaTeX thesis template for FPT University graduation projects. This directory is the **embedded thesis template** for the `mars-rec-sys` research codebase; it is tracked in git as part of the deliverable but is not run as part of the experiment workflow.

## Project Structure

```
thesis/
├── main.tex                 # Main document - clean entry point
├── preamble.tex            # All packages and formatting settings
├── chapters/               # Chapter files
│   ├── 00_cover.tex           # Cover page
│   ├── 01_statement.tex        # Statement of Originality
│   ├── 02_abstract.tex         # Abstract
│   ├── 03_introduction.tex     # Introduction
│   ├── 04_background.tex       # Background
│   ├── 05_related_work.tex     # Related Work
│   ├── 06_methodology.tex      # Methodology
│   ├── 07_experiments.tex      # Experiments & Results
│   └── 08_conclusion.tex       # Conclusion
├── appendices/             # Appendices
│   └── appendix_a.tex         # Sample appendix
├── refs/                   # Bibliography files
│   ├── refs1.bib
│   └── refs2.bib
├── images/                 # Images and figures
│   └── Logo_FPT_Education.png  # FPT logo (required)
├── build/                  # Build output directory (auto-created)
│   └── main.pdf               # Generated PDF
└── .vscode/                # VS Code configuration
    └── settings.json          # LaTeX Workshop settings
```

## Thesis Information

- **Title:** A gSASRec-Based Sequential Recommender System for Next Learning Resource Prediction on the MARS E-Learning Dataset
- **Supervisor:** Dr. Nguyễn An Khương
- **Class:** AI18C
- **Course:** AIL303m
- **Students:**
  - Nguyễn Hoàng Việt -- QE180058
  - Nguyễn Tấn Thắng -- QE180019
  - Lê Quốc Chính -- QE170250

## Building the PDF

### Prerequisites

You need to have the following installed:
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- `latexmk` (usually included with LaTeX)
- `biber` (for bibliography management)

### Using VS Code with LaTeX Workshop

1. Install the **LaTeX Workshop** extension by James Yu
2. Open this folder in VS Code
3. The `.vscode/settings.json` is automatically configured
4. Edit any `.tex` file and save - the PDF will build automatically
5. View the PDF in VS Code's integrated viewer (PDF tab opens automatically)

### Building from Command Line

```bash
cd thesis
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf -outdir=build main.tex
```

For a clean rebuild:
```bash
latexmk -C -outdir=build
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf -outdir=build main.tex
```

## Output

- **PDF Location:** `thesis/build/main.pdf`
- All intermediate build files (`.log`, `.aux`, `.bbl`, etc.) are stored in the `build/` directory
- The source files remain clean and uncluttered

## Features

- ✅ Clean separation of concerns (preamble, chapters, appendices)
- ✅ Professional formatting suitable for academic theses
- ✅ biblatex with biber backend for modern bibliography management
- ✅ Proper page margins (3cm left, 2cm others)
- ✅ 1.3 line spacing for readability
- ✅ Automatic PDF generation on save (VS Code)
- ✅ All build artifacts separated in `build/` directory
- ✅ Only one `\end{document}` at the very end
- ✅ Modular chapter structure for easy editing

## Key Configuration

### `.vscode/settings.json`
- Uses `latexmk` for compilation
- Output directory: `build/`
- PDF viewer: VS Code tab
- Auto-build on save: enabled
- Recipe: "latexmk with Biber" for bibliography

### `preamble.tex`
Contains all packages and formatting:
- Input encoding: UTF-8
- Languages: English and Vietnamese
- Paper: A4, 13pt font
- Margins: 3cm left, 2cm right/top/bottom
- Bibliography: biblatex with biber backend
- Line spacing: 1.3

## Editing Tips

1. **Main.tex:** Only edit the structure (which chapters to include). Keep it clean!
2. **Preamble.tex:** Add packages or modify formatting here
3. **Chapters:** Edit individual chapter files (00_cover.tex through 08_conclusion.tex)
4. **Bibliography:** Add references to refs1.bib and refs2.bib

## Adding New Content

### Add a new chapter:
1. Create `chapters/09_newchapter.tex`
2. Add `\input{chapters/09_newchapter}` to main.tex

### Add bibliography entries:
Simply add BibTeX entries to `refs/refs1.bib` or `refs/refs2.bib`

### Add images:
Place images in the `images/` directory and reference them:
```latex
\includegraphics[width=8cm]{images/image_name.png}
```

## Troubleshooting

### PDF won't build
- Check that `build/` directory exists (should auto-create)
- Ensure all `.tex` files are valid LaTeX
- Check the build log: `build/main.log`

### Bibliography not showing
- Run `biber` explicitly: `biber build/main`
- Then recompile: `latexmk -pdf -outdir=build main.tex`

### Images not found
- Ensure image file is in `images/` directory
- Use relative paths only: `{images/filename.png}`

## Notes

- This template uses the `article` document class (not `book`)
- All generated files are in the `build/` directory
- The thesis title and student information are set in `chapters/00_cover.tex`
- Bibliography is managed with biblatex (modern replacement for BibTeX)
