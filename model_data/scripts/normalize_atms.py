import os
import argparse
import zarr
from sklearn.preprocessing import StandardScaler

def normalize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input zarr file')
    parser.add_argument('output', type=str, help='Output zarr file')
    args = parser.parse_args()

    fields_names = ['sensorZenithAngle', 'solarAzimuthAngle', 'bt_channel_1']

    # copy input zarr file to output zarr file
    os.system(f'cp {args.input} {args.output}')

    # open zarr file and apply normalization to the fields, then save the modified data the output zarr file
    zarr_file = zarr.open(args.output, mode='a')
    for field in fields_names:
        data = zarr_file[field][:]
        zarr_file[field][:] = normalize(data[:])

    zarr_file.save()
    zarr_file.close()
