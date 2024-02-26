# atac_to_dnase

Model to convert ATAC signal to DNase signal

Convert DNase/ATAC signal to bigwig files

Installation
- clone repo
- install conda environment
- `pip install -e .`  # this is so files in scripts/ run correctly

Generate training regions
Splits ABC regions into 250bp regions (You can modify 250bp by editing atac_to_dnase/utils.py)
```
py scripts/gen_train_data.py --abc_regions data/raw/ABC_peaks.bed --output_file data/processed/training_regions.tsv
py main.py save_data --training_regions data/processed/training_regions.tsv --atac_bw data/raw/atac_ENCFF512VEZ.bigWig --dnase_bw data/raw/dnase_ENCFF860XAE.bigWig --fasta data/reference/hg38.fa
```


Generate RPM coverage for peak region
```
py scripts/RPM_coverage.py --regions data/processed/training_regions.tsv --atac_bw data/raw/atac_ENCFF512VEZ.bigWig --dnase_bw data/raw/dnase_ENCFF860XAE.bigWig --output_atac data/processed/atac_RPM_coverages.bedgraph --output_dnase data/processed/dnase_RPM_coverages.bedgraph
```

Compute Log2 Fold Change between DNase and ATAC signals
```
py scripts/plot_log_fold_change.py --rpm data/processed/region_RPM_coverages.tsv --crispr_file data/raw/EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz --output_file results/plots.pdf
```


Train the model
```
py main.py train --epochs 1000 --loss_plot train_2_head_8_blocks.pdf
```

Make predictions
```
py main.py predict --regions data/processed/training_regions.tsv --atac_bw data/raw/atac_ENCFF512VEZ.bigWig --fasta data/reference/hg38.fa --output_bw data/results/predicted_dnase.bigWig
```

