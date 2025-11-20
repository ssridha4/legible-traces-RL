import pickle
import os

def read_traces(pickle_path, num_samples=20):
    """
    Read a pickle file containing trace data and display the first N data points.
    
    Args:
        pickle_path: Path to the pickle file
        num_samples: Number of data points to display (default: 20)
    """
    # Load the pickle file
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    
    # Display metadata
    print("=" * 80)
    print("METADATA")
    print("=" * 80)
    if "metadata" in data:
        for key, value in data["metadata"].items():
            print(f"{key}: {value}")
    
    # Display data points
    print("\n" + "=" * 80)
    print(f"FIRST {num_samples} DATA POINTS")
    print("=" * 80)
    
    if "data" in data:
        data_dict = data["data"]
        
        # Get the keys from data dictionary
        keys = list(data_dict.keys())
        num_items = len(data_dict[keys[0]]) if keys else 0
        num_to_display = min(num_samples, num_items)
        
        for i in range(num_to_display):
            print(f"\n--- Data Point {i+1} ---")
            for key in keys:
                value = data_dict[key][i]
                # Truncate long strings for readability
                # if isinstance(value, str) and len(value) > 500:
                #     value = value[:500] + "... [truncated]"
                print(f"{key}: {value}")
    
    print("\n" + "=" * 80)
    print(f"Total data points in file: {num_items if 'data' in data and data['data'] else 0}")
    print("=" * 80)

if __name__ == "__main__":
    # Path to the pickle file
    # Go up 2 levels from src/utils/read_data.py to reach root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pickle_file = os.path.join(
        root_dir,
        "outputs",
        "traces",
        "gsm8k",
        "test",
        "traces_Qwen_Qwen3-0.6B_no_reasoning.pkl"
    )
    
    # Read and display first 20 data points
    read_traces(pickle_file, num_samples=3)

