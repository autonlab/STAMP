cd /path/to/benchmark_data/ISRUC/files

# Run 8 downloads at a time (tune -P based on your bandwidth)
seq 1 100 | xargs -n 1 -P 8 -I {} wget -c --show-progress \
"http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI/{}.rar"