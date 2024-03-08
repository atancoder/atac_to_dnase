# atac_to_dnase

Model to convert ATAC signal to DNase signal

Convert DNase/ATAC signal to bigwig files

Installation
- clone repo
- install conda environment
- `pip install -e .`  # this is so files in scripts/ run correctly

Requirements
- BigWig files (Convert from BAM file and normalized to sequencing depth)
- fasta file 
- ABC candidate regions bed file (remove chrM)

##Generate training regions
Splits ABC regions into 500bp regions and extends each side by 125bp 
Adds ATAC signal, DNase signal, and DNA sequence to each region
	- filter out regions that don't have both dnase and atac signal
	- filter out regions in unmappable regions (no sequence)
	
```
py main.py gen_regions --abc_regions data/raw/ABC_peaks.bed --region_size 500 --region_slop 125 --atac_bw data/raw/atac_ENCFF512VEZ.bigWig --dnase_bw data/raw/dnase_ENCFF860XAE.bigWig --fasta data/reference/hg38.fa --output_file data/processed/regions.tsv
```

##Train the model
Find optimal learning rate
```
py main.py lr_grid_search --regions data/processed/regions.tsv
```
```
py main.py train --regions data/processed/regions.tsv --saved_model models/model.pt --epochs 100 --loss_plot loss_plot.pdf
```

Make predictions
Regions must utilize the same region_size and region_slop that the model was trained on
TODO: Allow us to specify the regions we wish to predict in bed format; then the script verifies it matches what the model was trained on
		and we can load the saved param we used for REGION slop based on the model
```
py main.py predict --regions data/processed/regions.tsv --saved_model models/model.pt --atac_bw data/raw/atac_ENCFF512VEZ.bigWig --fasta data/reference/hg38.fa --output_folder results/
```

Evaluation:
Plot DNase vs ATAC signals
```
py scripts/plot_atac_vs_dnase.py --abc_regions data/raw/ABC_peaks.bed --atac_bw data/raw/atac_ENCFF512VEZ.bigWig --dnase_bw data/raw/dnase_ENCFF860XAE.bigWig --crispr_file data/raw/EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz --output_file results/plots.pdf
py scripts/plot_atac_vs_dnase.py --abc_regions data/raw/ABC_peaks.bed --atac_bw /oak/stanford/groups/engreitz/Users/atan5133/data/ENCODE/K562_DNASE/ENCFF414OGC.bigWig --dnase_bw /oak/stanford/groups/engreitz/Users/atan5133/data/ENCODE/K562_DNASE/no_scale.bigwig --crispr_file data/raw/EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz --output_file results/plot_same_exp_diff_replicates.pdf

```

