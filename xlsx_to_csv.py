import pandas as pd
import os
import csv

import settings


def csv_from_excel():
    print(f"csv from excel")
    data_xls = pd.read_excel(os.path.join(settings.data_folder, settings.source_file),
                             settings.sheet_name, index_col=None)
    data_xls.to_csv(os.path.join(settings.data_folder, settings.csv_name), encoding='utf-8', index=False)

