import zipfile
import os

for date in ['20251216','20251217', '20251218', '20251219', '20251222']:
    input_zip = rf"D:\trade\data\option\usa\second\fdx\{date}_iv_trade_american.zip"
    output_zip = rf"D:\trade\data\option\usa\second\fdx\{date}_iv_trade_american2.zip"

    with zipfile.ZipFile(input_zip, 'r') as zin:
        with zipfile.ZipFile(output_zip, 'w') as zout:
            for item in zin.infolist():
                # Apply your renaming logic here

                new_nm = os.path.basename(item.filename.replace('second_american', 'second_iv_trade_american'))

                # Copy raw compressed data directly
                zout.writestr(new_nm, zin.read(item.filename))

    os.remove(input_zip)
    os.rename(output_zip, input_zip)
