# PFAS Adsorbent Search

## Setup

### For fetching the data

#### Installing the required packages
conda env create -f environment.yaml
conda activate pfas

#### Updating the environment.yaml files

conda env update -f environment.yml --prune

## Scripts 

### Running the script to fetch data
Fetch the data from the PACE ICE cluster from the directory `/storage/ice-shared/cs8903onl/mussmann-pfas/data/` and copied into the `data/` directory.

```
cd pfas-environment-cleanup
python3 scripts/fetch_data.py
```