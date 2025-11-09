cd /path/to/benchmark_data/ISRUC/files/rar

for f in *.rar; do
    unar -o ../extracted "$f" # unrar did not work for the .rec files, so use unar instead
done