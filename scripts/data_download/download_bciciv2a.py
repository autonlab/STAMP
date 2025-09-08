

# Run wget https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat
import os
import wget
def download_bciciv2a_data():
    file_names = [
        "A01T.mat",
        "A01E.mat",
        "A02T.mat",
        "A02E.mat",
        "A03T.mat",
        "A03E.mat",
        "A04T.mat",
        "A04E.mat",
        "A05T.mat",
        "A05E.mat",
        "A06T.mat",
        "A06E.mat",
        "A07T.mat",
        "A07E.mat",
        "A08T.mat",
        "A08E.mat",
        "A09T.mat",
        "A09E.mat",
    ]
    for file_name in file_names:
        url = f"https://bnci-horizon-2020.eu/database/data-sets/001-2014/{file_name}"
        output_path = "/path/to/benchmark_data/BCICIV_2a/mat/" + file_name

        if not os.path.exists(output_path):
            print(f"Downloading data from {url}...")
            wget.download(url, out=output_path)
            print("\nDownload complete.")
        else:
            print("Data file already exists.")


if __name__ == "__main__":
    download_bciciv2a_data()