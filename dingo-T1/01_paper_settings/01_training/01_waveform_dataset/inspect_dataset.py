import h5py

f = h5py.File("waveform_dataset.hdf5","r")

print(f["parameters"].dtype.names)