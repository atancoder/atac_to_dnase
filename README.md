# atac_to_dnase

Model to convert ATAC signal to DNase signal
- Model consists of convolutional layers with a transformer architecture and uses relative positional encodings

Convert DNase/ATAC signal to bigwig files

Installation
- clone repo
- install conda environment
- `pip install -e .`  # this is so files in scripts/ run correctly

Requirements
- BigWig files (Convert from BAM file and normalized to sequencing depth)
- fasta file 
- ABC candidate regions bed file (remove chrM)

## Generate training regions

Splits ABC regions into 500bp regions and extends each side by 125bp 
Adds ATAC signal, DNase signal, and DNA sequence to each region
	- filter out regions that don't have both dnase and atac signal
	- filter out regions in unmappable regions (no sequence)
	
```
py main.py gen_regions --abc_regions data/raw/ABC_peaks.bed --region_size 500 --region_slop 125 --output_file data/processed/regions.tsv
```

## Train the model

Find optimal learning rate
```
py main.py lr_grid_search --chrom chr1 --regions data/processed/regions.tsv --atac_bw data/raw/ENCFF534DCE.bigWig --dnase_bw data/raw/ENCFF338LXW.bigWig --fasta data/reference/hg38.fa
```
```
py main.py train --regions data/processed/regions.tsv --saved_model models/model.pt --chrom chr1 --epochs 1000 --atac_bw data/raw/ENCFF534DCE.bigWig --dnase_bw data/raw/ENCFF338LXW.bigWig --fasta data/reference/hg38.fa --loss_plot plots/rel_pos_simple_model.pdf
```

## Validate the Model

```
py main.py validate --regions data/processed/regions.tsv --saved_model models/model.pt --chrom chr2 --atac_bw data/raw/ENCFF534DCE.bigWig --dnase_bw data/raw/ENCFF338LXW.bigWig --fasta data/reference/hg38.fa
```

Make predictions
Regions must utilize the same region_size and region_slop that the model was trained on
TODO: Allow us to specify the regions we wish to predict in bed format; then the script verifies it matches what the model was trained on
		and we can load the saved param we used for REGION slop based on the model
```
py main.py predict --regions data/processed/regions.tsv --saved_model models/model.pt --atac_bw data/raw/ENCFF534DCE.bigWig --fasta data/reference/hg38.fa --output_folder results/
```

Evaluation:
Get ATAC/DNase signal at peaks
```
py scripts/gen_region_signal.py --regions data/processed/regions.tsv --atac_bw data/raw/ENCFF534DCE.bigWig --dnase_bw data/raw/ENCFF338LXW.bigWig --output_file data/processed/coverage_signal.tsv
```

Plot DNase vs ATAC signals
```
py scripts/plot_atac_vs_dnase.py --abc_regions data/raw/ABC_peaks.bed --atac_bw data/raw/ENCFF534DCE.bigWig --dnase_bw data/raw/ENCFF338LXW.bigWig --crispr_file data/raw/EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz --output_file results/plots.pdf
```

