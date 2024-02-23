# atac_to_dnase

Model to convert ATAC signal to DNase signal

Convert DNase/ATAC signal to bigwig files

Installation
- clone repo
- install conda environment
- `pip install -e .`

Generate training regions
1. Splits ABC regions into 250bp regions 
2. Attach DNA sequence to each region
```
py scripts/gen_train_data.py --abc_regions data/raw/ABC_peaks.bed --fasta reference/hg38.fa --output_file data/processed/training_regions.tsv
```

Generate RPM coverage for peak region
```
py scripts/RPM_coverage.py --regions data/processed/training_regions.tsv --atac_bw data/raw/atac_ENCFF512VEZ.bigWig --dnase_bw data/raw/dnase_ENCFF860XAE.bigWig --output_file data/processed/region_RPM_coverages.tsv
```

Compute Log2 Fold Change between DNase and ATAC signals
```
py scripts/plot_log_fold_change.py --rpm data/processed/region_RPM_coverages.tsv --crispr_file data/raw/EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz --output_file results/plots.pdf
```


Train the model
```
py main.py --training_regions data/processed/training_regions.tsv --atac_bw data/raw/atac_ENCFF512VEZ.bigWig --dnase_bw data/raw/dnase_ENCFF860XAE.bigWig
```