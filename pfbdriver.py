
import sys

import numpy as np

import pfb
import h5py
from drift.util import mpiutil

fname_in = sys.argv[1]
nfreq = int(sys.argv[2])
fname_out = sys.argv[3]
ntap = int(sys.argv[4])

pfb_data = np.load(fname_in).reshape(-1, nfreq)

nblock = pfb_data.shape[0]

local_block, start_block, end_block = mpiutil.split_local(nblock)

pfb_local = pfb_data[start_block:end_block]

rects = pfb.inverse_pfb_parallel(pfb_local, ntap, nblock)

local_tsblock, start_tsblock, end_tsblock = mpiutil.split_local(nblock)

for ri in range(mpiutil.size):
    if mpiutil.rank == ri:

        with h5py.File(fname_out, 'a') as f:
            if 'ts' not in f:
                f.create_dataset('ts', shape=(nblock, rects.shape[-1]), dtype=rects.dtype)

            f['ts'][start_tsblock:end_tsblock] = rects

    mpiutil.barrier()
