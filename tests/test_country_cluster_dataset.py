from datasets.country_cluster_rice_dataset import CountryClusterRiceDataset
from pathlib import Path


def test_country_cluster_rice_dataset():
    root = Path("../data_to_analysis/rice/countries")
    dataset = CountryClusterRiceDataset(root=root)

    for d in dataset:
        print(d)
    return


if __name__ == "__main__":
    test_country_cluster_rice_dataset()