import cv2
import os

test_path = "data/processed/t2i_images/pair_0002/simple/pair_0002_simple_001_seed1000.png"
print("Dateigröße in Bytes:", os.path.getsize(test_path))

img = cv2.imread(test_path)
print("Konnte OpenCV das Bild lesen?", img is not None)
