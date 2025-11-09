import torch
import torch.multiprocessing as mp
import os
from tqdm import tqdm
import subprocess
import datetime
import json
import gc
import socket
from torch.utils.data import DataLoader, Subset
from tsfm_public.models.tspulse.utils.helpers import get_embeddings
import numpy as np
import time
import shutil
from stamp.datasets.utils import get_simple_gpu_info
from stamp.datasets.lmdb_writer import LMDBWriter

def build_embeddings(dataset, available_gpus, chunk_size, temp_chunks_dir,
                                         model_name, model_dir, model_load_fn, use_amp,
                                         output_path, map_size_per_mode, embedding_agg):
    if model_load_fn is None or not callable(model_load_fn):
        raise ValueError("model_load_fn must be a callable function")

    if not available_gpus:
        raise ValueError("available_gpus cannot be empty")

    validate_gpus(available_gpus)

    start_time = time.time()
    print(f"Starting distributed embedding on {len(available_gpus)} GPUs: {available_gpus}")
    print(f"Temporary chunks will be saved to: {temp_chunks_dir}")

    for mode in ['train', 'val', 'test']:
        print(f"Processing mode: {mode}")

        map_size = map_size_per_mode[mode]

        # Create temporary output directory for this mode's chunks
        mode_chunks_dir = os.path.join(temp_chunks_dir, mode)
        os.makedirs(mode_chunks_dir, exist_ok=True)

        # Process the dataset across available GPUs
        _ = embed_simple_multi_gpu(
            dataset=dataset,
            mode=mode,
            available_gpus=available_gpus,
            chunk_size=chunk_size,
            output_dir=mode_chunks_dir,
            use_amp=use_amp,
            model_name=model_name,
            model_dir=model_dir,
            model_load_fn=model_load_fn,
            embedding_agg=embedding_agg
        )

        print(f"Merging {mode} results to LMDB...")
        merge_distributed_chunks(
            output_dir=mode_chunks_dir,
            split_name=mode,
            output_path=output_path + f'/{mode}',
            cleanup_chunks=True,
            map_size=map_size
        )

    print(f"Distributed embedding complete!")
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

    if os.path.exists(temp_chunks_dir):
        shutil.rmtree(temp_chunks_dir)
        print(f"Cleaned up temporary chunks directory: {temp_chunks_dir}")

    # Save configuration
    gpu_info = get_simple_gpu_info(available_gpus)
    params = {
        'model_name': model_name,
        'model_dir': model_dir,
        'batch_size': dataset.params.batch_size,
        'preprocessed_dataset_dir': dataset.dataset_dir,
        'chunk_size': chunk_size,
        'output_path': output_path,
        'map_size_per_mode': map_size_per_mode,
        'run_time': total_time,
        'available_gpus': available_gpus,
        'gpu_info': gpu_info,
    }

    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        params['commit_hash'] = commit_hash
    except subprocess.CalledProcessError as e:
        print(f'{e} The directory is not a git repo so the commit hash could not be retrieved.')

    params['run_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save config next to HDF5 file
    config_path = output_path + f'/config.json'
    with open(config_path, 'w') as f:
        json.dump(params, f, indent=4)

    return None

def embed_simple_multi_gpu(dataset, mode, available_gpus,
                          chunk_size, output_dir, use_amp,
                          model_name, model_dir, model_load_fn,
                          embedding_agg):
    """Simple multi-GPU processing without distributed context"""

    world_size = len(available_gpus)
    print(f"Starting simple multi-GPU processing on {world_size} GPUs: {available_gpus}")
    print(f"Mode: {mode}")

    os.makedirs(output_dir, exist_ok=True)

    if world_size > 1:
        # Use multiprocessing but without distributed context
        mp.spawn(
            embed_distributed_worker,
            args=(available_gpus, dataset, mode,
                  chunk_size, output_dir, model_name, model_dir,
                  model_load_fn, use_amp, embedding_agg),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU
        embed_distributed_worker(
            0, available_gpus, dataset, mode,
            chunk_size, output_dir, model_name, model_dir, 
            model_load_fn, use_amp, embedding_agg
        )

    print(f"Multi-GPU processing complete! Results saved to {output_dir}")
    return output_dir

def embed_distributed_worker(rank, available_gpus, dataset, mode,
                                   chunk_size, output_dir, model_name, model_dir, model_load_fn, use_amp, embedding_agg):
    """Simplified worker function without distributed context"""

    batch_size = dataset.params.batch_size
    n_temporal_channels = dataset.n_temporal_channels
    n_spatial_channels = dataset.n_spatial_channels

    gpu_id = int(available_gpus[rank])
    device = torch.device(f'cuda:{gpu_id}')

    print(f"Rank {rank}: Starting on GPU {gpu_id}")

    # Load model directly on GPU
    model = model_load_fn(model_name, model_dir, device=f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    
    if 'chronos' not in model_name.lower():
        # Only do this for non-Chronos models, device moving is handled in load_chronos_model
        model = model.to(device)
        model.eval()

    # Create data loader for this rank (without distributed sampling)
    # You'll need to manually split the data
    total_gpus = len(available_gpus)

    # Get the full dataset
    data_loaders = dataset.get_data_loader()  # Non-distributed version

    if mode not in data_loaders:
        raise ValueError(f"Mode '{mode}' not found. Available modes: {list(data_loaders.keys())}")

    full_dataloader = data_loaders[mode]

    collate_fn = full_dataloader.collate_fn

    # Manually partition data for this GPU
    dataset_size = len(full_dataloader.dataset)

    # Calculate exact start and end indices for this rank
    samples_per_gpu = dataset_size // total_gpus
    remainder = dataset_size % total_gpus

    start_idx = rank * samples_per_gpu + min(rank, remainder)
    end_idx = start_idx + samples_per_gpu + (1 if rank < remainder else 0)

    indices = np.arange(start_idx, end_idx)
    gpu_dataset = Subset(full_dataloader.dataset, indices)

    gpu_dataloader = DataLoader(
        gpu_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0, # Single-threaded data loading to prevent pickle failures "Memo value not found"
        collate_fn=collate_fn
    )

    # Create rank-specific output directory
    rank_output_dir = os.path.join(output_dir, f'rank_{rank}')
    os.makedirs(rank_output_dir, exist_ok=True)

    print(f"GPU {gpu_id}: Processing samples {indices[0]} to {indices[-1]} samples)")

    # Process embeddings for this rank
    embeddings_output = embed_single_gpu_worker(
        data_loader=gpu_dataloader,
        batch_size=batch_size,
        n_temporal_channels=n_temporal_channels,
        n_spatial_channels=n_spatial_channels,
        model=model,
        model_name=model_name,
        device=device,
        output_dir=rank_output_dir,
        chunk_size=chunk_size // total_gpus,
        use_amp=use_amp,
        rank=rank,
        gpu_id=gpu_id,
        world_size=total_gpus,
        embedding_agg=embedding_agg
    )

    print(f"Rank {rank}: Completed processing on GPU {gpu_id}")
    return embeddings_output

def embed_single_gpu_worker(data_loader, batch_size, n_temporal_channels, n_spatial_channels,
                            model, model_name, device, output_dir, chunk_size, use_amp, rank, gpu_id, world_size, embedding_agg):
    """Process embeddings on a single GPU"""

    # Process batches
    all_embeddings = []
    all_labels = []
    all_sample_keys = []
    processed_samples = 0
    chunk_idx = 0

    is_chronos = 'chronos' in model_name.lower()
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"GPU {gpu_id} (rank {rank})", position=rank) if world_size > 1 else tqdm(data_loader, desc="Processing")

        for batch_idx, batch in enumerate(pbar):
            
            if is_chronos:
                x_data, y_label, sample_keys = batch
                mask = None
            else:
                x_data, y_label, mask, sample_keys = batch

            # For MOMENT:
                # x_data shape: (batch_size * n_spatial_channels * n_temporal_channels, 1, seq_len)
                # mask shape: (batch_size * n_spatial_channels * n_temporal_channels, seq_len)
            # For TSPulse:
                # x_data shape: (batch_size * n_spatial_channels * n_temporal_channels, seq_len, 1)
                # mask shape: (batch_size * n_spatial_channels * n_temporal_channels, seq_len, 1)
            # For Chronos:
                # x_data shape: (batch_size * n_spatial_channels * n_temporal_channels, seq_len)
                # No mask
            x_data = x_data.to(device, non_blocking=True) 
            if not is_chronos:
                mask = mask.to(device, non_blocking=True) 

            # Forward pass
            if 'moment' in model_name.lower():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(x_enc=x_data, input_mask=mask)
                else:
                    outputs = model(x_enc=x_data, input_mask=mask)
                embeddings = outputs.embeddings.cpu() # Shape: (batch_size * n_spatial_channels * n_temporal_channels, embedding_dim)
            elif 'tspulse' in model_name.lower():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = get_embeddings(model, past_values=x_data, past_observed_mask=mask, mode="register")
                else:
                    outputs = get_embeddings(model, past_values=x_data, past_observed_mask=mask, mode="register")

                # Reshape outputs from (batch_size * n_spatial_channels * n_temporal_channels, 1, embedding_dim) to (batch_size * n_spatial_channels * n_temporal_channels, embedding_dim)
                embeddings = outputs.squeeze(1).contiguous().cpu() # Shape: (batch_size * n_spatial_channels * n_temporal_channels, embedding_dim)
            elif 'chronos' in model_name.lower():
                # Chronos expects input shape (batch_size, seq_len) and outputs embeddings of shape (batch_size, seq_len + 1, embedding_dim)
                with torch.device(device): # NOTE: This is important! For some reason, Chronos' tokenizer creates tensors on CPU by default
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            embeddings, _ = model.embed(x_data) # Shape: (batch_size * n_spatial_channels * n_temporal_channels, seq_len + 1, embedding_dim)
                    else:
                        embeddings, _ = model.embed(x_data) # Shape: (batch_size * n_spatial_channels * n_temporal_channels, seq_len + 1, embedding_dim)
                
                embeddings = embeddings.cpu()

                if embedding_agg == 'eos':
                    # Use the embedding of the EOS token
                    embeddings = embeddings[:, -1, :] # Shape: (batch_size * n_spatial_channels * n_temporal_channels, embedding_dim)
                elif embedding_agg == 'mean':
                    # Remove the EOS token and average over the sequence dimension
                    embeddings = embeddings[:, :-1, :].mean(dim=1) # Shape: (batch_size * n_spatial_channels * n_temporal_channels, embedding_dim)
                else:
                    raise ValueError(f"Invalid embedding_agg method '{embedding_agg}' for Chronos model. Use 'eos' or 'mean'.")

                embeddings = embeddings.contiguous() # NOTE: Important for view() later, otherwise may get "RuntimeError: view size is not compatible with input tensor's size and stride"
            else:
                raise ValueError(f"Model name '{model_name}' not recognized for embedding extraction")
                
            batch_size = x_data.shape[0] // (n_temporal_channels * n_spatial_channels)
            embeddings = embeddings.view(batch_size, -1)

            # The lmdb_pickle_dataset.py has sample_keys as dictionary containing metadata so we have to handle this
            if isinstance(sample_keys[0], dict):
                stride = n_spatial_channels * n_temporal_channels
                sample_keys = [sample_keys[i * stride]['sample_key'] for i in range(batch_size)]

            # Store results
            all_embeddings.append(embeddings)
            all_labels.append(y_label.cpu())
            all_sample_keys.extend(sample_keys)

            processed_samples += batch_size

            # Save chunk if needed
            if processed_samples >= chunk_size:
                save_embedding_chunk_distributed(
                    embeddings_list=all_embeddings, labels_list=all_labels, sample_keys_list=all_sample_keys,
                    output_dir=output_dir, chunk_idx=chunk_idx, rank=rank, model_name=model_name
                )

                # Clear memory
                all_embeddings.clear()
                all_labels.clear()
                all_sample_keys.clear()
                chunk_idx += 1
                processed_samples = 0

                gc.collect()
                torch.cuda.empty_cache()

            # Periodic cleanup
            if batch_idx % 1000 == 0:
                torch.cuda.empty_cache()

    # Save final chunk
    if all_embeddings:
        save_embedding_chunk_distributed(
            embeddings_list=all_embeddings, labels_list=all_labels, sample_keys_list=all_sample_keys,
            output_dir=output_dir, chunk_idx=chunk_idx, rank=rank, model_name=model_name
        )

    return output_dir

def save_embedding_chunk_distributed(embeddings_list, labels_list, sample_keys_list,
                                   output_dir, chunk_idx, rank, model_name):
    """Save embeddings chunk for distributed processing"""
    chunk_embeddings = torch.cat(embeddings_list, dim=0) # (total_samples, full_dim)
    chunk_labels = torch.cat(labels_list, dim=0)

    base_filename = f'chunk_rank{rank}_{chunk_idx:04d}'

    # Save as separate NPY files for optimal performance
    embeddings_file = os.path.join(output_dir, f'{base_filename}_embeddings.npy')
    labels_file = os.path.join(output_dir, f'{base_filename}_labels.npy')
    keys_file = os.path.join(output_dir, f'{base_filename}_keys.npy')

    if 'chronos' in model_name.lower():
        # Chronos outputs bfloat16 by default, convert to float32 for compatibility
        chunk_embeddings = chunk_embeddings.to(torch.float32).numpy()
    else:
        chunk_embeddings = chunk_embeddings.numpy().astype(np.float32)

    np.save(embeddings_file, chunk_embeddings)
    np.save(labels_file, chunk_labels.numpy())
    sample_keys_array = np.array(sample_keys_list, dtype=object)
    np.save(keys_file, sample_keys_array)

    if rank == 0:  # Reduce print spam
        sequences = len(chunk_embeddings)
        samples = sequences // 160 if sequences > 160 else sequences
        print(f"GPU {rank} saved chunk {chunk_idx}: {sequences:,} sequences ({samples:,} samples)")

def merge_distributed_chunks(output_dir, split_name, output_path, cleanup_chunks, map_size):
    """
    Merge distributed chunks from npy files and save to LMDB format

    Args:
        output_dir: Directory containing rank_* subdirectories
        split_name: Name of the split ('train', 'val', 'test')
        n_temporal_channels: Number of temporal channels
        n_spatial_channels: Number of spatial channels
        output_path: Path to save LMDB file
        cleanup_chunks: Whether to clean up temporary chunk files
        map_size: Maximum size of LMDB database in bytes
    """
    print("Merging distributed chunks to HDF5...")

    # Find all rank directories
    rank_dirs = [d for d in os.listdir(output_dir) if d.startswith('rank_')]
    rank_dirs.sort()

    if not rank_dirs:
        raise ValueError(f"No rank directories found in {output_dir}")

    # Create LMDB database
    with LMDBWriter(output_path, map_size=map_size) as writer:
        total_chunks = 0
        sample_idx = 0

        for rank_dir in rank_dirs:
            rank_path = os.path.join(output_dir, rank_dir)

            # Find unique chunk bases (look for _embeddings.npy files)
            chunk_bases = set()
            for f in os.listdir(rank_path):
                if f.endswith('_embeddings.npy'):
                    base = f.replace('_embeddings.npy', '')
                    chunk_bases.add(base)

            chunk_bases = sorted(list(chunk_bases))
            total_chunks += len(chunk_bases)

            print(f"Loading {len(chunk_bases)} chunks from {rank_dir}")

            for chunk_base in tqdm(chunk_bases, desc=f"Loading {rank_dir}"):
                embeddings_file = os.path.join(rank_path, f'{chunk_base}_embeddings.npy')
                labels_file = os.path.join(rank_path, f'{chunk_base}_labels.npy')
                keys_file = os.path.join(rank_path, f'{chunk_base}_keys.npy')

                # Load NPY files (much faster than pickle)
                embeddings = np.load(embeddings_file)  # Shape: (N, embedding_dim)
                labels = np.load(labels_file)
                sample_keys = np.load(keys_file, allow_pickle=True)  # Object array of strings

                # Store each embedding individually in LMDB
                for embedding, label, sample_key in zip(embeddings, labels, sample_keys):
                    # Use the sample_key directly as the filename
                    # Clean the sample_key to ensure it's a valid filename
                    clean_sample_key = str(sample_key).replace('\x00', '').strip()
                    if not clean_sample_key:
                        clean_sample_key = f"{split_name}_sample_{sample_idx}"

                    # Store embedding with label in key format
                    writer.write_sample(embedding, int(label), clean_sample_key, dtype=np.float32)
                    sample_idx += 1

    print(f"Merged {sample_idx:,} embeddings from {total_chunks} chunks across {len(rank_dirs)} GPUs")
    print(f"Saved to LMDB: {output_path}")

    # Cleanup chunks if requested
    if cleanup_chunks:
        cleanup_distributed_chunks(output_dir, keep_merged=False)

    return output_path

def cleanup_distributed_chunks(output_dir, keep_merged=False):
    """Clean up temporary chunk files"""
    print(f"Cleaning up chunks in {output_dir}")

    if not keep_merged and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed temporary directory: {output_dir}")

def find_free_port():
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def validate_gpus(available_gpus):
    """Validate that all requested GPUs exist"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    max_gpu = torch.cuda.device_count() - 1
    for gpu_id in available_gpus:
        if gpu_id > max_gpu:
            raise ValueError(f"GPU {gpu_id} does not exist. Available GPUs: 0-{max_gpu}")

    return True