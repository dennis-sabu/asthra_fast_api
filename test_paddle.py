from paddleocr import PaddleOCRVL
import time

IMAGE_PATH = r"C:\Users\denni\OneDrive\Documents\vs code\websites\asthra_fast_api\id.jpeg"

print("1. Loading PaddleOCR-VL...")

pipeline = PaddleOCRVL(
    pipeline_version="v1.6"
)

print("2. Model loaded!")
print("3. Processing ID card...")
print("Image:", IMAGE_PATH)

start = time.time()

output = pipeline.predict(IMAGE_PATH)

elapsed = time.time() - start

print(f"\n4. Processing finished in {elapsed:.2f} seconds")
print("\n========== RESULT ==========\n")

for res in output:
    res.print()

print("\n========== END ==========")