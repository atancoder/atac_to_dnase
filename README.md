# atac_to_dnase

Model to convert ATAC signal to DNase signal

Generate 500bp regions
```
py workflow/scripts/create_fixed_size_regions.py --abc_regions data/ABC_peaks.bed --output_file data/500bp_peak_regions.bed
```

Compute coverage (in RPM) for each of these regions
```
py workflow/scripts/RPM_coverage.py --peak_regions data/500bp_peak_regions.bed --profile data/ENCFF860XAE.sorted.se.bam --chrom_sizes data/GRCh38_EBV.chrom.sizes.tsv --output_file data/regions_dhs_coverage.bed
py workflow/scripts/RPM_coverage.py --peak_regions data/500bp_peak_regions.bed --profile data/xu_K562_sorted.tagAlign.gz --chrom_sizes data/GRCh38_EBV.chrom.sizes.tsv --output_file data/regions_atac_coverage.bed

```

Compute Log2 Fold Change between DHS and ATAC signals
```
py workflow/scripts/plot_log_fold_change.py --dhs data/regions_dhs_coverage.bed --atac data/regions_atac_coverage.bed --crispr_file data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz  --output plots.pdf

```