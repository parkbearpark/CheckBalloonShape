import argparse
from argparse import RawTextHelpFormatter
import cv2
import os
from BalloonDetector import detectFromPaths
from PanelExtractor import PanelExtractor


# フキダシの輪郭を検出する
def detect_contours(image, path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # 二値化する
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

    # 輪郭を抽出する
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 吹き出しの輪郭を特定する
    bubble_contour = None
    for i in range(len(contours)):
        # 輪郭の階層構造をチェックして、親輪郭が存在する場合は吹き出しの輪郭と判断する
        if hierarchy[0][i][3] != -1:
            bubble_contour = contours[i]
            break

    print(bubble_contour)

    # 吹き出しの輪郭を近似する
    epsilon = 0.1 * cv2.arcLength(bubble_contour, True)
    approx = cv2.approxPolyDP(bubble_contour, epsilon, True)

    # 近似された輪郭の形状を分析して、フキダシがギザギザなのか丸っぽいものなのかを判定する
    if len(approx) == 4:
        print('ギザギザ')
    else:
        print('丸っぽい')

    return image, contours


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
    contour_detected_output_dir = f'{args.folder}/contour_detected'

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(balloon_detected_output_dir):
        os.mkdir(balloon_detected_output_dir)

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
