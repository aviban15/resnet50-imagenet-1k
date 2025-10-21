import torch

def show_gpu_info():
    if not torch.cuda.is_available():
        print("No GPU detected. Using CPU.")
        return

    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):        
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1024**3  # Convert bytes to GB
        print(f"\nGPU {i}: {props.name}")
        print(f"  - Total Memory: {total_mem:.2f} GB")
        print(f"  - Compute Capability: {props.major}.{props.minor}")

if __name__ == "__main__":
    show_gpu_info()