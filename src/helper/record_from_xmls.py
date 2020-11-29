import argparse
import os
import pandas as pd
import tensorflow as tf

from csv_from_xmls import csv_from_df, df_from_xmls
from object_detection.utils import dataset_util, label_map_util
from PIL import Image
from six import BytesIO
from typing import Dict, List, TypedDict


# reduce log
# log_util.reduce_log()  # TODO: move script to root so that "from detector.util import log_util" works
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().propagate = False


# typing
class ImgLiteral(TypedDict):
    fn: str
    objs_df: pd.DataFrame


def create_example(label_dict: Dict, in_dir: str, sub_dir: str, img_literal: ImgLiteral) -> tf.train.Example:
    fn = img_literal['fn']
    objs_df = img_literal['objs_df']
    
    img_data = tf.io.gfile.GFile(os.path.join(in_dir, sub_dir, '{}'.format(fn)), 'rb').read()
    img = Image.open(BytesIO(img_data))
    w, h = img.size
    
    fn = fn.encode('utf8')
    img_format = b'jpg'
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    cls_texts = []
    cls_labels = []
    
    for i, row in objs_df.iterrows():
        cls = row['class']
        xmins.append(row['xmin'] / w)
        ymins.append(row['ymin'] / h)
        xmaxs.append(row['xmax'] / w)
        ymaxs.append(row['ymax'] / h)
        cls_texts.append(cls.encode('utf8'))
        cls_labels.append(label_dict[cls])
    
    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(h),
        'image/width': dataset_util.int64_feature(w),
        'image/filename': dataset_util.bytes_feature(fn),
        'image/source_id': dataset_util.bytes_feature(fn),
        'image/encoded': dataset_util.bytes_feature(img_data),
        'image/format': dataset_util.bytes_feature(img_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(cls_texts),
        'image/object/class/label': dataset_util.int64_list_feature(cls_labels),
    }))


def create_record(label_dict: Dict, in_dir: str, out_dir: str, sub_dir: str, out_csv: bool) -> None:
    df = df_from_xmls(in_dir, sub_dir)
    if df is None:
        return
    
    if out_csv:
        csv_from_df(in_dir, sub_dir, df)
    
    writer = tf.io.TFRecordWriter(os.path.join(out_dir, sub_dir, '_' + sub_dir + '.tfrecord'))
    print('creating TFRecord for {} set'.format(sub_dir))
    img_literals = group_example_by_img(df)
    
    for img_literal in img_literals:
        example = create_example(label_dict, in_dir, sub_dir, img_literal)
        writer.write(example.SerializeToString())
    writer.close()


def create_records(label_path: str, in_dir: str, out_dir: str, out_csv: bool) -> None:
    label_map = label_map_util.load_labelmap(label_path)
    label_dict = label_map_util.get_label_map_dict(label_map)
    
    for sub_dir in ['train', 'dev', 'test']:
        create_record(label_dict, in_dir, out_dir, sub_dir, out_csv)


def group_example_by_img(df: pd.DataFrame) -> List[ImgLiteral]:
    gb = df.groupby('filename')
    
    return [{'fn': fn, 'objs_df': gb.get_group(group_name)} for fn, group_name in zip(gb.groups.keys(), gb.groups)]


def main() -> None:
    # argument parser
    parser = argparse.ArgumentParser(
        description='convert xml files to TensorFlow TFRecord',
    )
    parser.add_argument(
        '-l', '--label_path',
        help='path to label file (.pbtxt)',
        type=str,
    )
    parser.add_argument(
        '-i', '--in_dir',
        help='input folder containing train/dev/test sub-folders (default to current working directory)',
        type=str,
        default=os.getcwd(),
    )
    parser.add_argument(
        '-o', '--out_dir',
        help='output folder for train/dev/test sets (default to be the same as input folder)',
        type=str,
        default=None,
    )
    parser.add_argument(
        '-c', '--out_csv',
        help='whether to output csv file',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    
    # dynamic default value
    if args.out_dir is None:
        args.out_dir = args.in_dir
    
    # proceed
    create_records(args.label_path, args.in_dir, args.out_dir, args.out_csv)


if __name__ == '__main__':
    main()
