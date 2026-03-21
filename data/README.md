# Data Directory

Raw data files are not committed to this repository.
Run the download script to populate this directory:

    bash scripts/download_datasets.sh

## What gets downloaded

| Folder | Contents | Size |
|--------|----------|------|
| raw/Eco1C1G1T1.UCF.json | Cello E. coli gate library | ~500KB |
| raw/ecoli_k12.fna | E. coli K-12 genome (BLAST db) | ~4.6MB |
| reference/ | iGEM characterized parts (fetched via BioPython) | ~50MB |

## Data Sources
- Cello UCF: github.com/CIDAR-LAB/Cello-v2
- E. coli K-12: NCBI Accession U00096 / GCF_000005845.2
- iGEM Registry: parts.igem.org
