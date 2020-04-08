import argparse
import os

from datasets import breta, camb, cvl, iam, orand

location = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(location, "../../data/raw/")
datasets = {
    "breta": [breta.extract, os.path.join(data_folder, "breta"), 1],
    "iam": [iam.extract, os.path.join(data_folder, "iam"), 2],
    "cvl": [cvl.extract, os.path.join(data_folder, "cvl"), 3],
    "orand": [orand.extract, os.path.join(data_folder, "orand"), 4],
    "camb": [camb.extract, os.path.join(data_folder, "camb"), 5],
    "all": [],
}

output_folder = "words_final"


parser = argparse.ArgumentParser(
    description="Script extracting words from raw dataset."
)
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    choices=datasets.keys(),
    help="Pick dataset(s) to be used.",
)
parser.add_argument(
    "-p",
    "--path",
    nargs="*",
    default=[],
    help="""Path to folder containing the dataset. For multiple datasets
    provide path or ''. If not filled, default paths will be used.""",
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset == ["all"]:
        args.dataset = list(datasets.keys())[:-1]

    assert args.path == [] or len(args.dataset) == len(
        args.path
    ), "provide same number of paths as datasets (use '' for default)"
    if args.path != []:
        for ds, path in zip(args.dataset, args.path):
            datasets[ds][1] = path

    for ds in args.dataset:
        print("Processing -", ds)
        entry = datasets[ds]
        entry[0](entry[1], output_folder, entry[2])
