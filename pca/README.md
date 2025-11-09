# PCA Application Figures

This directory contains (or will contain) the four figures for the "Real-World Applications" slide (page 4).

## Quick Start - Generate All Images

The easiest way to get all four images is to run the Python script:

```bash
cd figures/
python generate_pca_figures.py
```

This will create:
- `eigenfaces.png`
- `genomics_pca.png`
- `financial_pca.png`
- `mnist_pca.png`

**Requirements:**
```bash
pip install numpy matplotlib scikit-learn
```

---

## Alternative: Download Existing CC-Licensed Images

If you prefer to download existing images instead of generating them, here are some options:

### 1. Eigenfaces (`eigenfaces.png`)

**Option A - Wikipedia/Wikimedia Commons:**
- Visit: https://en.wikipedia.org/wiki/Eigenface
- Click on images in the article (they're hosted on Wikimedia Commons)
- Download the eigenfaces visualization
- Save as `eigenfaces.png`

**Option B - Scikit-learn Documentation:**
- Visit: https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
- Screenshot or download the eigenfaces grid
- Save as `eigenfaces.png`

**License:** Wikipedia images are typically CC-BY-SA or public domain

### 2. Genomics PCA (`genomics_pca.png`)

**Option A - PMC Open Access Articles:**
- Visit: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5976774/
- Look for PCA scatter plot figures (Figure 2 or similar)
- Download under CC-BY 4.0 license
- Save as `genomics_pca.png`

**Option B - Harvard Bioinformatics Core:**
- Visit: https://hbctraining.github.io/scRNA-seq/lessons/05_normalization_and_PCA.html
- Find PCA scatter plot examples
- Save as `genomics_pca.png`

**License:** CC-BY 4.0

### 3. Financial PCA (`financial_pca.png`)

**Option A - PLOS One (Open Access):**
- Visit: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0230124
- Download figures showing stock price PCA
- Save as `financial_pca.png`

**Option B - Generate from real data:**
Use the Python script (recommended) or:
- Download stock data from Yahoo Finance (free, no auth required)
- Create visualization manually

**License:** CC-BY 4.0 (PLOS One)

### 4. MNIST PCA (`mnist_pca.png`)

**Option A - Colah's Blog:**
- Visit: https://colah.github.io/posts/2014-10-Visualizing-MNIST/
- Screenshot the PCA visualization
- Save as `mnist_pca.png`

**Option B - GitHub:**
- Search GitHub for "MNIST PCA visualization"
- Look for repositories with MIT or CC licenses
- Many have pre-generated images you can use

**License:** Varies by source (check individual repos)

---

## File Naming Convention (IMPORTANT!)

The LaTeX file expects these EXACT filenames:

1. **eigenfaces.png** - Face recognition eigenfaces grid
2. **genomics_pca.png** - Gene expression PCA scatter plot
3. **financial_pca.png** - Stock price time series + PCA
4. **mnist_pca.png** - MNIST digits with principal components

If you download images from other sources, **rename them to match these filenames exactly** for plug-and-play compatibility with the LaTeX document.

---

## Image Specifications

For best results, your images should:
- Be at least 1200x900 pixels (for good quality at slide resolution)
- PNG format (supports transparency)
- Have clear labels and captions
- Use readable fonts (size 10-12pt minimum)

---

## License Information

If you generate images using the Python script:
- The code is provided for educational use
- Generated figures use publicly available datasets (Olivetti faces, MNIST)
- Safe to use in academic presentations

If you download images:
- Check the license of each source
- CC-BY and CC-BY-SA require attribution
- Include attribution in your presentation if required
- Most academic figures from PMC and PLOS are CC-BY 4.0

---

## Troubleshooting

**Script fails to run:**
- Make sure you have Python 3.7+ installed
- Install required packages: `pip install numpy matplotlib scikit-learn`
- On first run, scikit-learn will download datasets (~5-10 MB)

**Images look wrong in LaTeX:**
- Check that filenames match exactly (case-sensitive)
- Ensure images are in the `figures/` directory
- Recompile LaTeX document: `pdflatex lecture.tex`

**Need different image dimensions:**
- Edit the `figsize` parameters in `generate_pca_figures.py`
- Standard is (8, 6) or (10, 6) for individual panels
