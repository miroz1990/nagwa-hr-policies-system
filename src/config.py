import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER = os.path.join(BASE_DIR, "data", "hr_policies")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "chunks.json")