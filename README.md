# ATAC to DNase
Sequence based model that converts ATAC signal to DNase signal.

## Model Info
This is a transformer based model that encodes ATAC signal + DNA sequence and predicts the corresponding
DNase signal track at base pair resolution.

![image](https://github.com/atancoder/atac_to_dnase/assets/10254642/e0263c43-b5c9-48c4-8b18-b5b3a0c0b6b9)


## Usage
Installation
- clone repo
- install conda environment (`mamba env create -f envs/env.yml`)
- `pip install -e .`  # this is so files in scripts/ run correctly

File Requirements
- [ATAC]([url](https://www.encodeproject.org/files/ENCFF093IIW/)) + [DNase]([url](https://www.encodeproject.org/files/ENCFF338LXW/)) BigWig files
- [fasta file]([url](https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/)) 
- [ABC]([url](https://github.com/broadinstitute/ABC-Enhancer-Gene-Prediction)) candidate regions bed file (remove chrM)

### 1) Generate training regions

Splits ABC regions into 500bp regions and extends each side by 125bp 
Adds ATAC signal, DNase signal, and DNA sequence to each region
	- filter out regions that don't have both dnase and atac signal
	- filter out regions in unmappable regions (no sequence)
	
```
py main.py gen_regions --abc_regions data/raw/ABC_peaks.bed --region_size 500 --region_slop 125 --output_file data/processed/regions.tsv
```

### 2) Train the model

Find optimal learning rate
```
py main.py lr_grid_search --chrom chr1 --regions data/processed/regions.tsv --atac_bw data/raw/ENCFF534DCE.bigWig --dnase_bw data/raw/ENCFF338LXW.bigWig --fasta data/reference/hg38.fa
```

Train the model on a GPU
- Train on a subset of chromosomes so that we can cross validate the model with other chromosomes later
```
py main.py train --regions data/processed/regions.tsv --saved_model models/model.pt --chrom chr1 --epochs 1000 --atac_bw data/raw/ENCFF534DCE.bigWig --dnase_bw data/raw/ENCFF338LXW.bigWig --fasta data/reference/hg38.fa --loss_plot plots/rel_pos_simple_model.pdf
```

Validate the Model
```
py main.py validate --regions data/processed/regions.tsv --saved_model models/model.pt --chrom chr2 --atac_bw data/raw/ENCFF534DCE.bigWig --dnase_bw data/raw/ENCFF338LXW.bigWig --fasta data/reference/hg38.fa
```

### 3) Make predictions
Note: Regions must utilize the same region_size and region_slop that the model was trained on
```
py main.py predict --regions data/processed/regions.tsv --saved_model models/model.pt --atac_bw data/raw/ENCFF534DCE.bigWig --fasta data/reference/hg38.fa --output_folder results/
```

Output folder will have the predicted DNase signal in bigWig format.

### 4) Visualizing the Predictions
Looking at loss metrics, such as MSE (mean squared error), may not be that informative. We need to compare the performance of the predicted track with the original track.

Get ATAC and DNase signal at peaks
```
py scripts/gen_region_signal.py --regions data/processed/regions.tsv --atac_bw data/raw/ENCFF534DCE.bigWig --dnase_bw data/raw/ENCFF338LXW.bigWig --output_file data/processed/coverage_signal.tsv
```

Plot DNase vs ATAC signals
```
py scripts/plot_atac_vs_dnase.py --abc_regions data/raw/ABC_peaks.bed --atac_bw data/raw/ENCFF534DCE.bigWig --dnase_bw data/raw/ENCFF338LXW.bigWig --crispr_file data/raw/EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz --output_file results/plots.pdf
```

Then replace the DNase signal with the predicted signal to see how the predictions fare.

### Results
