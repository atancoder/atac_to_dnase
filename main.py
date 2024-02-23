import click

from atac_to_dnase.data import get_dataloader


@click.command
@click.option("--training_regions", required=True)
@click.option("--atac_bw", required=True)
@click.option("--dnase_bw", required=True)
def main(training_regions, atac_bw, dnase_bw):
    data_loader = get_dataloader(training_regions, atac_bw, dnase_bw)


if __name__ == "__main__":
    main()
