import argparse
from argparse import RawTextHelpFormatter
import cv2
import os
from BalloonDetector import detectFromPaths
from PanelExtractor import PanelExtractor


def main(args):
    print('Extracting panels...')

    # コマの画像ファイルを抽出
    panel_extractor = PanelExtractor(keep_text=True)
    panel_extractor.extract(args.folder)

    print('Extracting panels... Done!')

    # コマの画像ファイルがあるディレクトリ
    input_dir = f'{args.folder}/panels'
    # 画像を保存するディレクトリ
    balloon_detected_output_dir = f'{args.folder}/balloons'
    text_removed_output_dir = f'{args.folder}/text_removed'

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(balloon_detected_output_dir):
        os.mkdir(balloon_detected_output_dir)
    if not os.path.exists(text_removed_output_dir):
        os.mkdir(text_removed_output_dir)

    print('Detecting balloons...')

    # ディレクトリ内の画像ファイルのパスを取得
    input_paths = [os.path.join(input_dir, filename)
                   for filename in os.listdir(input_dir)]

    # フキダシを検出して切り取り、画像を保存
    images, locations, new_paths = detectFromPaths(
        input_paths)
    for i in range(len(images)):
        image = images[i]
        _new_path = new_paths[i].replace('.jpg', '')
        new_path = f'{_new_path}_{str(i)}.jpg'
        cv2.imwrite(os.path.join(
            balloon_detected_output_dir, os.path.basename(new_path)), image)
    print('Detecting balloons... Done!')

    print('Removing text...')

    text_removed_imgs = panel_extractor.remove_text(images)
    for i in range(len(text_removed_imgs)):
        image = images[i]
        _new_path = new_paths[i].replace('.jpg', '')
        new_path = f'{_new_path}_{str(i)}.jpg'
        cv2.imwrite(os.path.join(
            text_removed_output_dir, os.path.basename(new_path)), image)

    print('Removing text... Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Implementation of a Manga Panel Extractor and dialogue bubble text eraser.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-f", '--folder', default='./input_images/', type=str,
                        help="""folder path to input manga pages.
Panels will be saved to a directory named `panels` in this folder.""")
    args = parser.parse_args()
    main(args)
