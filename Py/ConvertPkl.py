# -*- coding: utf-8 -*-
# One-time conversion: splits HPC-generated combined pkl into
# numpy-only pkl + DataFrames xlsx (avoids pandas version issues).
import pickle, sys
import pandas as pd

def convert(srcPkl, arraysPkl, metaXlsx):
    data = pickle.load(open(srcPkl, 'rb'))
    print(f'Loaded {srcPkl}  keys: {list(data.keys())}')

    pickle.dump({'Timecourses': data['Timecourses'],
                 'parcelsSize': data['parcelsSize']},
                open(arraysPkl, 'wb'))
    print(f'Saved arrays → {arraysPkl}')

    with pd.ExcelWriter(metaXlsx, engine='openpyxl') as writer:
        data['Runs'].to_excel(  writer, sheet_name='Runs',    index=False)
        data['Labels'].to_excel(writer, sheet_name='Labels',  index=False)
    print(f'Saved metadata → {metaXlsx}')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python ConvertPkl.py <src.pkl> <arrays.pkl> <meta.xlsx>')
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
