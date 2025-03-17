from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as et
from argparse import ArgumentParser


def extract_from_xml(file: Path):
    out = []
    tree = et.parse(file)
    for img in tree.findall('.//image'):
        box = img.find('.//box')
        plate = box.find('.//attribute')
        tmp = img.attrib | box.attrib | {'plate_number': plate.text}
        out.append(tmp)
    return out


def convert_to_xc_yc_w_h(data: dict) -> tuple[float, float, float, float]:
    img_height = float(data['height'])
    img_width = float(data['width'])
    xtl = float(data['xtl'])
    ytl = float(data['ytl'])
    xbr = float(data['xbr'])
    ybr = float(data['ybr'])

    width = xbr - xtl
    height = ybr - ytl
    x_center = (xtl + (width / 2)) / img_width
    y_center = (ytl + (height / 2)) / img_height

    round_to_n = lambda x: round(x, 4)  # for n = 4
    return round_to_n(x_center), round_to_n(y_center), round_to_n(width / img_width), round_to_n(height / img_height)



def move_file_and_label(data: dict, img_dir: Path, image_to: Path, label_to: Path):
    file = img_dir / data['name']
    shutil.copy(file, image_to / file.name)
    xc, yc, w, h = convert_to_xc_yc_w_h(data)
    with open(label_to / (str(file.stem) + '.txt'), 'w') as f:
        f.write(f"0 {xc} {yc} {w} {h}\n")



def main():
    parser = ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Path to the dataset directory")
    parser.add_argument("destination", type=Path, help="Path to save the output")
    parser.add_argument("--divide", type=float, default=0.8, help="Ratio to split dataset (default: 0.8)")
    parser.add_argument("--seed", type=int, default=-1, help="Optional random seed for reproducibility")
    args = parser.parse_args()

    image_path: Path = args.dataset.resolve() / 'photos'
    annotations_path: Path = args.dataset.resolve() / 'annotations.xml'

    data = extract_from_xml(annotations_path)

    # split the dataset
    if args.seed != -1:
        random.seed(args.seed)
    random.shuffle(data)
    divide_index = int(len(data) * args.divide)
    train = data[:divide_index]
    val = data[divide_index:]

    # make new directory
    new_dir: Path = args.destination.resolve()
    new_images_dir = new_dir / 'images'
    new_labels_dir = new_dir / 'labels'

    if not new_dir.exists():
        raise FileNotFoundError(f"Directory {str(new_dir)} does not exist.")

    for path in (new_images_dir, new_images_dir / 'train', new_images_dir / 'val',
                 new_labels_dir, new_labels_dir / 'train', new_labels_dir / 'val'):
        if path.exists():
            raise FileExistsError(f"Directory {str(path)} already exists.")
        path.mkdir()

    for img_data in train:
        move_file_and_label(img_data, image_path, new_images_dir / 'train', new_labels_dir / 'train')

    for img_data in val:
        move_file_and_label(img_data, image_path, new_images_dir / 'val', new_labels_dir / 'val')

    yaml_file_contents = f'''
train: {str((new_images_dir / 'train').absolute())}
val: {str((new_images_dir / 'val').absolute())}

nc: 1
names: ["license_plate"]
    '''.strip()

    with open(new_dir / 'license_plate.yaml', 'w') as f:
        f.write(yaml_file_contents)


if __name__ == '__main__':
    main()
