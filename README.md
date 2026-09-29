# Fluorescence USAF MTF Analyzer

A desktop GUI for quantitative analysis of fluorescent **USAF 1951 resolution targets**. The program extracts averaged intensity profiles from selected bar-pattern regions, calculates Michelson contrast transfer (CTF), compares horizontal and vertical responses, and exports publication-ready figures.

The application was developed for fluorescence microscopy and MUSE system characterization, but it can be used with any grayscale or RGB USAF target image.

## Features

- TIFF, PNG, JPG, and BMP input
- Packed `uint32` RGB32 TIFF support (`0x00RRGGBB`)
- Gray, red, green, and blue channel analysis
- 0°, 90°, 180°, and 270° image rotation
- Interactive ROI selection
- ROI select, deselect, delete, and clear controls
- Averaged 1-D profiles for vertical and horizontal bar sets
- Automatic peak and valley detection
- Michelson CTF calculation
- Horizontal and vertical transfer curves
- Replicate ROI averaging with mean ± SD in exported figures
- CSV measurement export
- Fixed-size scientific figure export in PNG, TIFF, PDF, and SVG

## Scientific definitions

For USAF 1951 group `G` and element `E`, the spatial frequency is

```text
f = 2^(G + (E - 1)/6)      [line pairs/mm]
```

The line-pair period is

```text
period = 1000 / f           [µm]
```

and the width of one bright or dark bar is

```text
bar width = 500 / f         [µm]
```

Michelson contrast is calculated from the averaged bar maxima and minima:

```text
CTF = (Imax - Imin) / (Imax + Imin)
```

The normalized transfer curve is the CTF divided by the lowest measured spatial-frequency CTF for the same bar orientation.

> **Note:** A USAF bar target directly provides discrete contrast-transfer measurements. The normalized curve produced here is a bar-target transfer estimate and should not be described as an ISO slanted-edge MTF measurement.

## Installation

The code was tested with a Miniforge/Conda Python environment.

```bash
conda activate accesspath
python -m pip install -r requirements.txt
```

Tkinter is required for the GUI and is included with most standard Python/Miniforge installations on Windows.

## Run

```bash
python mtf_gui.py
```

## Recommended workflow

1. Load the **native-resolution** fluorescence image.
2. Select the analysis channel.
3. Rotate the image if required.
4. Enter the USAF group and element.
5. Choose the correct bar orientation:
   - **Vertical bars** → X-direction response
   - **Horizontal bars** → Y-direction response
6. Set the object-plane pixel size if known.
7. For a standard three-bar USAF element, start with:
   - Peaks to average: `3`
   - Smoothing sigma: `0.3–0.7 px`
8. Draw an ROI around one complete bar element.
9. Inspect the profile and detected peaks/valleys.
10. Repeat for additional USAF elements and both orientations.
11. Export measurements and the scientific figure.

## Publication figure

The scientific export uses a fixed physical canvas so figures from different datasets remain the same size. The output includes:

- full image with analysis ROIs
- enlarged selected ROI
- selected ROI intensity profile
- absolute Michelson CTF curve
- normalized transfer curve

Files are saved as:

```text
USAF_Fig.png
USAF_Fig.tif
USAF_Fig.pdf
USAF_Fig.svg
USAF_Fig_caption.txt
```

PNG and TIFF are saved at 600 dpi. PDF and SVG preserve vector text and plots.

## Input image recommendations

For quantitative analysis, avoid screenshots, PowerPoint exports, JPEG compression, or resized images. Use the original camera image whenever possible. Frame averaging is acceptable if the same acquisition/averaging procedure is used across conditions being compared.

## Project structure

```text
.
├── mtf_gui.py
├── README.md
├── requirements.txt
└── .gitignore
```


## Citation

If you use this software in a publication, presentation, thesis, or other research output, please cite the software repository.

**Suggested citation**

> Karim, M. M. (2026). *Fluorescence USAF MTF Analyzer* (Version 1.0.0) [Computer software]. GitHub. https://github.com/Mobarak-Karim/USAF-MTF-Analyzer

A machine-readable citation is also provided in `CITATION.cff`. GitHub can use this file to display a **Cite this repository** option.

When reporting results generated with this program, also describe the image source, pixel size, USAF group/element, bar orientation, ROI selection procedure, smoothing setting, number of peaks/valleys used, and whether the reported quantity is absolute Michelson CTF or normalized CTF.

## License

No license has been assigned yet. Add an appropriate license before public redistribution if needed.