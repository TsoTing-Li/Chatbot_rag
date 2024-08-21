import os
from os import walk

import fitz


def is_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        if doc.page_count > 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def repair_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        new_file_path = "repaired_" + file_path
        doc.save(new_file_path)
        print(f"Repaired file saved as {new_file_path}")
    except Exception as e:
        print(f"Error: {e}")


def pdf_handler(dir: str):
    if not os.path.exists(dir):
        raise FileExistsError(f"'{dir}' is not exist!")
    for root, folder, files in walk(dir):
        for file in files:
            pdf = os.path.join(root, file)

            if not is_pdf(pdf):
                print(pdf)
                repair_pdf(pdf)
    print("Finish File(PDF) check")
