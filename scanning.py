import os
import time
import struct
from PIL import Image
from pyfprint import Fprint

# Ensure that libfprint is installed in your system
def capture_fingerprint():
    with Fprint() as f:
        devices = f.discover_devices()
        if not devices:
            print("No fingerprint scanner found!")
            return None
        
        dev = devices[0]  # Select first available scanner
        dev.open()

        print("Place your finger on the scanner...")
        img = dev.enroll_finger()

        if img:
            print("Fingerprint captured successfully!")
            return img
        else:
            print("Failed to capture fingerprint.")
            return None

def save_fingerprint_as_bmp(image_data, filename="fingerprint.bmp"):
    img = Image.fromarray(image_data)
    img.save(filename, format="BMP")
    print(f"Fingerprint image saved as {filename}")

if __name__ == "__main__":
    fingerprint_img = capture_fingerprint()
    if fingerprint_img:
        save_fingerprint_as_bmp(fingerprint_img)
