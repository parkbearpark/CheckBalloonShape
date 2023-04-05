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
    # フキダシを囲む矩形を描画した画像を保存するディレクトリ
    output_dir = f'{args.folder}/balloons'

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Detecting balloons...')

    # ディレクトリ内の画像ファイルのパスを取得
    input_paths = [os.path.join(input_dir, filename)
                   for filename in os.listdir(input_dir)]

    # フキダシを検出して矩形で囲み、画像を保存
    images, locations, new_paths = detectFromPaths(
        input_paths)
    for i in range(len(images)):
        image = images[i]
        location = locations[i]
        _new_path = new_paths[i].replace('.jpg', '')
        new_path = f'{_new_path}_{str(i)}.jpg'
        cv2.imwrite(os.path.join(
            output_dir, os.path.basename(new_path)), image)

    print('Detecting balloons... Done!')


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
