import os
import zipfile
import io
import pandas as pd
from shared.constants import file_root


def gen_openinterest_files(sec_type, market, resolution, symbol, start_date):
    if resolution == 'daily':
        path = os.path.join(file_root, sec_type, market, resolution)
    else:
        path = os.path.join(file_root, sec_type, market, resolution, symbol)

    for directory, subdirectories, files in os.walk(path):
        for fn in files:  # Open the ZIP archive in read mode
            if start_date and resolution in ('tick', 'minute', 'second') and fn[:8] < start_date:
                continue
            path_zip_new = os.path.join(path, fn.replace('quote', 'openinterest'))
            if fn.endswith('_quote_american.zip') and not os.path.exists(path_zip_new) and not 'iv_quote' in fn:
                print(fn)
                with zipfile.ZipFile(os.path.join(path, fn), 'r') as archive:
                    # Get a list of all the member filenames in the archive
                    # Create a new ZIP archive in write mode
                    with zipfile.ZipFile(path_zip_new, 'w') as new_archive:
                        # Iterate over each filename in the list
                        # Otherwise, add the member to the new archive
                        for fn_csv in archive.namelist():
                            member_csv_new = fn_csv.replace('quote', 'openinterest')
                            if member_csv_new in new_archive.namelist():
                                print(f'Skipping {member_csv_new}. Already exists.')
                                continue
                            if resolution == 'daily':
                                df = pd.read_csv(io.BytesIO(archive.read(fn_csv)))
                                if not df.empty:
                                    df.iloc[:, 1] = 0
                                    new_archive.writestr(member_csv_new, df.iloc[:, [0, 1]].to_csv(header=False, index=False))
                                    print(f'Wrote zip entry {path_zip_new}#{member_csv_new}')
                            elif resolution in ['minute', 'hour', 'second', 'tick']:
                                new_archive.writestr(member_csv_new, '23400000,0\n')
                                print(f'Wrote zip entry {path_zip_new}#{member_csv_new}')


if __name__ == '__main__':
    start_date = '20231013'
    end_date = '20241231'
    for resolution in ['tick', 'minute', 'second', 'daily']:
        for sym in ['CSCO', 'DELL', 'ORCL', 'PFE', 'HPE', 'IPG', 'AKAM', 'AOS', 'A', 'MO', 'FL', 'ALL', 'ARE', 'ZBRA', 'AES', 'APD', 'ALLE', 'LNT', 'ZTS', 'ZBH']:
            gen_openinterest_files(sec_type='option', market='usa', resolution=resolution, symbol=sym, start_date=start_date)
    # gen_openinterest_files('option', 'usa', 'minute', 'hpe')
    # sec_type='option'; market='usa'; resolution='daily'; symbol='hpe'
