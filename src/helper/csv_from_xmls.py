import argparse
import glob
import os
import pandas as pd

from typing import Optional
from xml.etree import ElementTree


def csv_from_df(in_dir: str, sub_dir: str, df: pd.DataFrame) -> None:
    df.to_csv(os.path.join(in_dir, sub_dir, '_' + sub_dir + '.csv'), index=None)


def csv_from_xmls(in_dir: str) -> None:
    for sub_dir in ['train', 'dev', 'test']:
        df = df_from_xmls(in_dir, sub_dir)
        if df is None:
            continue
        
        csv_from_df(in_dir, sub_dir, df)


def df_from_xmls(in_dir: str, sub_dir: str) -> Optional[pd.DataFrame]:
    xml_dir = os.path.join(in_dir, sub_dir)
    if not os.path.isdir(xml_dir):
        print('directory "{}" not exist'.format(xml_dir))
        return

    xml_files = glob.glob(os.path.join(in_dir, sub_dir, '*.xml'))
    if len(xml_files) <= 0:
        print('xml file not found in directory "{}"'.format(xml_dir))
        return
    
    rows = []
    for xml_file in xml_files:
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        
        # fn = root.find('filename').text  # filename is not up to date after being copied
        ext = os.path.splitext(root.find('filename').text)[1]
        fn = os.path.splitext(os.path.basename(xml_file))[0] + ext
        
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        for obj in root.findall('object'):
            cls = obj.find('name').text
            box = obj.find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            
            rows.append((fn, w, h, cls, xmin, ymin, xmax, ymax))
    
    return pd.DataFrame(rows, columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])


def main() -> None:
    # argument parser
    parser = argparse.ArgumentParser(
        description='consolidate xml files into a csv file, for each of the sub-datasets',
    )
    parser.add_argument(
        '-i', '--in_dir',
        help='input folder containing train/dev/test sub-folders (default to current working directory)',
        type=str,
        default=os.getcwd(),
    )
    args = parser.parse_args()

    # proceed
    csv_from_xmls(args.in_dir)


if __name__ == '__main__':
    main()
