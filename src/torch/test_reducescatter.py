# allgather_example.py
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend='mpi')

    count = 3

    rank = dist.get_rank()
    size_ = dist.get_world_size()

    input_tensors = [torch.tensor([i] * count, dtype=torch.int16) for i in range(size_)]
    output_tensors = torch.zeros(count, dtype=torch.int16)

    print(f"Rank {rank} ReduceScatter input: {input_tensors}")
    dist.reduce_scatter(output_tensors, input_tensors)
    print(f"Rank {rank} ReduceScatterreceived: {output_tensors}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

 
