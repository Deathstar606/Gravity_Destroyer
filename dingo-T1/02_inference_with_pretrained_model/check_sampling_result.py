"""Check the sampling HDF5 to see what keys / posterior columns are available."""
import h5py, sys

path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/events/GW190701_203306/H1/result/GW190701_203306_data0_1246048404-56_sampling.hdf5"

def print_tree(f, prefix=""):
    for k in f.keys():
        item = f[k]
        if isinstance(item, h5py.Group):
            print(f"{prefix}[Group] {k}/")
            print_tree(item, prefix + "  ")
        else:
            shape = item.shape if hasattr(item, 'shape') else "?"
            dtype = item.dtype if hasattr(item, 'dtype') else "?"
            print(f"{prefix}[Dataset] {k}  shape={shape}  dtype={dtype}")

with h5py.File(path, "r") as f:
    print("Top-level keys:", list(f.keys()))
    print()
    print_tree(f)
