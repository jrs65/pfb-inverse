
import argparse

import numpy as np

import pfb
import h5py
import mpiutil

parser = argparse.ArgumentParser(description='Invert the PFB.')

parser.add_argument('file_in', help='Input file to process.')
parser.add_argument('file_out', help='File to write output timestream into.')
parser.add_argument('-f', type=int, default=1024, help='Number of frequencies in file.', dest='nfreq')
parser.add_argument('-n', type=int, default=4, help='Number of taps used for PFB', dest='ntap')
parser.add_argument('-m', type=int, action='store_true', help='Input file is missing Nyquist frequency.', dest='no_nyquist')

args = parser.parse_args()

print args



#=========================
#
# This is where we must load in the data. At the moment we have a very stupid method.
#
# At the end of it, each process must end up with a section of the file from
# start_block to end_block, and these must be as doubles.
#
#=========================

pfb_data = np.load(args.file_in).reshape(-1, args.nfreq)  # Load in whole file!

nblock = pfb_data.shape[0]  # Find out the file length in blocks
local_block, start_block, end_block = mpiutil.split_local(nblock)  # Work out how to split up the file into sections

pfb_local = pfb_data[start_block:end_block]  # Pull out the correct section of the file.

#=========================


# Apply inverse PFB
rects = pfb.inverse_pfb_parallel(pfb_local, args.ntap, nblock, no_nyquist=args.no_nyquist)

# Calculate the range of local timestream blocks
local_tsblock, start_tsblock, end_tsblock = mpiutil.split_local(nblock)



#=========================
#
# This is where we must output the data. Again this is a very stupid way of doing it.
#
#=========================
for ri in range(mpiutil.size):
    if mpiutil.rank == ri:

        with h5py.File(args.file_out, 'a') as f:
            if 'ts' not in f:
                f.create_dataset('ts', shape=(nblock, rects.shape[-1]), dtype=rects.dtype)

            f['ts'][start_tsblock:end_tsblock] = rects

    mpiutil.barrier()

#=========================