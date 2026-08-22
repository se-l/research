"""
My zip file has duplicate file names. Can you show me sample code how to remove duplicates in python using winzip and store the .zip with unique members only?
"""
import os
import shutil
import zipfile

from shared.paths import Paths

root = Paths.path_data.joinpath('option')


def rm_dups(trade_type):
    for directory, subdirectories, files in os.walk(root):
        for fn in files:  # Open the ZIP archive in read mode
            if trade_type not in fn:
                continue
            with zipfile.ZipFile(os.path.join(directory, fn), 'r') as archive:
                # Get a list of all the member filenames in the archive
                filenames = archive.namelist()
                new_archive_files = []
                # Create a new ZIP archive in write mode
                with zipfile.ZipFile(os.path.join(directory, f'uni_{fn}'), 'w') as new_archive:
                    # Iterate over each filename in the list
                    for filename in filenames:
                        # Check if the filename appears more than once in the list
                        if filename in new_archive_files:
                            print(f'Skipping {filename}')
                            continue
                        new_archive_files.append(filename)
                        # Otherwise, add the member to the new archive
                        content = archive.read(filename)
                        new_archive.writestr(filename, content)


def rn(trade_type):
    for directory, subdirectories, files in os.walk(root):
        for fn in files:  # Open the ZIP archive in read mode
            if trade_type not in fn:
                continue
            shutil.move(os.path.join(directory, fn), os.path.join(directory, fn.replace('uni_', '')))


if __name__ == '__main__':
    for trade_type in ['trade', 'quote', 'openinterest']:
        rm_dups(trade_type)
        rn(trade_type)
