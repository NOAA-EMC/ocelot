import os
import re
import argparse
import zarr
from sklearn.preprocessing import StandardScaler

def normalize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

def apply_normalization(output):
    zarr_file = zarr.open(output, mode='a')
    fields_names = ['sensorZenithAngle', 'solarAzimuthAngle']

    # add the bt_channel fields
    for key in zarr_file.keys():
        if key.startswith('bt_channel'):
            fields_names.append(key)

    for field in fields_names:
        data = zarr_file[field][:]
        zarr_file[field][:] = normalize(data)

    zarr_file.save()
    zarr_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input zarr file')
    args = parser.parse_args()

    output = re.sub(r'\.zarr$', '_normalized.zarr', args.input)

    # copy input zarr file to output zarr file
    os.system(f'cp -r {args.input} {output}')
    apply_normalization(output)
