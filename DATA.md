# Data availability and cohort definition

## Curated 33-image cohort

The manuscript analysis uses the 33 filenames listed in `configs/manuscript_33.csv`:

- HeLa: 20 images
- BxPC-3: 5 images
- MCF-7: 8 images

`Series041_B4.tif` is not part of the frozen 33-image cohort. During cross-model quality control, its MitoSegNet output contained one component occupying 783,992 pixels, corresponding to 74.8% of the field of view. The same image was therefore excluded from every method before the final cross-model analysis. This exclusion and its criterion must be reported consistently in the manuscript and supplementary information.

## Files not redistributed

The repository does not include:

- full-resolution CLSM study images;
- Cellpose, MitoSegNet, MoDL, Nellie, or Mitometer prediction sets;
- third-party model weights;
- patient or personally identifiable information.

The complete six-model analysis can be reproduced after placing the authorized prediction files under `predictions_33/<model>/` using filenames matching the manifest.

## Bundled example

`example_data/raw_CLSM.tif` is included only to verify installation and the P3 inference path. It is not one of the 33 study images and must not be used as an additional study observation.
