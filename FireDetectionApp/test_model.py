import cv2
import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU



from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'best.pt')


def main():
    parser = argparse.ArgumentParser(description='Run a quick inference test with the trained fire model.')
    parser.add_argument('--image', required=True, help='Path to an input image file')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')

    if not os.path.exists(args.image):
        raise FileNotFoundError(f'Image file not found: {args.image}')

    model = YOLO(MODEL_PATH)
    model.to('cpu')

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f'Failed to read image: {args.image}')

    results = model(img)[0]
    for det in results.boxes:
        cls_id = int(det.cls[0])
        cls_name = model.names[cls_id]
        print(f"{cls_name}: {det.conf[0]:.2f}")


if __name__ == '__main__':
    main()
