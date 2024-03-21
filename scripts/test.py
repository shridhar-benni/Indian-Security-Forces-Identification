import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--in_imgs_path',
        default="..\input_images\\",
        help='folder path containing image/images for detection.')
    args = parser.parse_args()
