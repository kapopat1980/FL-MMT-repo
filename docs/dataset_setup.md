# Dataset setup

All datasets used in this work are public and require no special access
beyond standard registration/agreement processes. This document lists
where to obtain each one and how it maps into the patient-level manifest
consumed by `src/data/dataset.py`.

## 1. LC25000 (histopathology)

- Source: Kaggle / Borkowski et al. (arXiv:1912.12142)
- Modality: lung and colon histopathology image patches
- Used for: histopathology stream pretraining and subtype classification

## 2. TCGA-LUAD / TCGA-LUSC (histopathology + genomic)

- Source: NCI GDC Data Portal (https://portal.gdc.cancer.gov)
- Modality: whole-slide images + somatic mutation calls
- Used for: histopathology + genomic streams, mutation-prediction auxiliary task
- Note: requires a free GDC account; no restricted-access (dbGaP) data is used

## 3. LIDC-IDRI (CT)

- Source: The Cancer Imaging Archive (TCIA)
- Modality: chest CT volumes with radiologist nodule annotations
- Used for: CT stream, pixel-level attribution validation (Integrated Gradients)

## 4. NLST (CT + clinical)

- Source: The Cancer Imaging Archive (TCIA) / National Lung Screening Trial
- Modality: low-dose CT + structured clinical/demographic data
- Used for: CT + clinical streams, longitudinal/screening cohort

## 5. IQ-OTH/NCCD (CT)

- Source: IEEE DataPort
- Modality: chest CT images (benign/malignant/normal)
- Used for: CT stream, external validation

## Preprocessing pipeline

Preprocessing scripts (not yet included in this initial release; being added
incrementally) will:

1. Convert DICOM/NIfTI volumes to a consistent voxel spacing and crop to
   the lung field.
2. Extract patch-level embeddings from histopathology WSIs using a
   DINO-pretrained ViT backbone (Caron et al., 2021).
3. Parse TCGA MAF files into per-patient mutation feature vectors.
4. Join all modalities into a single per-patient manifest (`data/manifest.csv`)
   with explicit null handling for patients missing one or more modalities.

## Building the manifest

```bash
python scripts/preprocess/build_manifest.py \
    --lc25000_dir /path/to/lc25000 \
    --tcga_dir /path/to/tcga \
    --lidc_dir /path/to/lidc_idri \
    --nlst_dir /path/to/nlst \
    --iqoth_dir /path/to/iqoth_nccd \
    --output data/manifest.csv
```

(This script is part of the ongoing code release — see the repository's
Issues tab for current status.)
