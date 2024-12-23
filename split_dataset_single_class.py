import os
import shutil
from pathlib import Path
import random
import hashlib


def check_and_clean_test_directory(base_dir):
    """
    Verifică dacă directorul `test` conține imagini și oferă opțiunea de a-l șterge,
    fără a afecta directoarele `train` și `validation`.
    """
    base_path = Path(base_dir)
    test_path = base_path / "test"  # Doar directorul "test"

    if test_path.exists() and any(test_path.rglob("*.*")):  # Verifică dacă există fișiere în subdirectoare
        choice = input(f"Directory '{test_path}' is not empty. Do you want to clean it? (yes/no): ").strip().lower()
        if choice == "yes":
            shutil.rmtree(test_path)  # Șterge întregul director
            print(f"Cleaned directory: {test_path}")
            test_path.mkdir(parents=True, exist_ok=True)  # Recrează directorul "test"
        else:
            print(f"Skipping cleaning for: {test_path}")
    elif not test_path.exists():
        test_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {test_path}")


def calculate_image_hash(image_path):
    """
    Calculează un hash unic (MD5) pentru o imagine.
    """
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def avoid_duplicates(image_path, destination_dir):
    """
    Verifică dacă imaginea există deja în directorul destinație, bazându-se pe hash-ul imaginii.
    """
    existing_hashes = set()
    for existing_image in Path(destination_dir).rglob("*.*"):
        existing_hashes.add(calculate_image_hash(existing_image))

    image_hash = calculate_image_hash(image_path)
    return image_hash not in existing_hashes  # Returnează True dacă imaginea este unică


def copy_images_with_no_duplicates(images, destination_dir):
    """
    Copiază imaginile în directorul destinație, evitând duplicatele.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        if avoid_duplicates(image, destination_dir):  # Verifică dacă imaginea este unică
            shutil.copy(image, destination_dir / image.name)
            print(f"Copied {image} to {destination_dir}")
        else:
            print(f"Duplicate image skipped: {image}")


def allocate_images_to_classes(source_dir, classes):
    """
    Alocă imaginile dintr-un director sursă în funcție de clase, pe baza numelor de fișiere.
    """
    source_path = Path(source_dir)
    images = list(source_path.glob("*.*"))

    if not images:
        raise ValueError(f"No images found in source directory: {source_dir}")

    allocation = {class_name: [] for class_name in classes}

    # Distribuție echilibrată a imaginilor pe clase
    class_count = len(classes)
    for i, image in enumerate(images):
        class_name = classes[i % class_count]  # Alocare ciclică pentru echilibrare
        allocation[class_name].append(image)

    # Verificare: afișăm numărul de imagini alocate fiecărei clase
    for class_name, class_images in allocation.items():
        print(f"Class '{class_name}' has {len(class_images)} images allocated.")

    return allocation


def split_dataset_test_only(source_dir, output_dir, source, classes, test_ratio=1):
    """
    Împarte imaginile doar în directorul `test`, fără a afecta `train` sau `validation`.
    """
    output_dir = Path(output_dir)

    # Verificare proporție
    if test_ratio <= 0 or test_ratio > 1:
        raise ValueError("Test ratio must be between 0 and 1.")

    # Curățarea directorului `test`, dacă este necesar
    check_and_clean_test_directory(output_dir)

    # Creează directoarele pentru `test`
    for class_name in classes:
        (output_dir / "test" / source / class_name).mkdir(parents=True, exist_ok=True)

    # Alocă imaginile pe clase
    allocation = allocate_images_to_classes(source_dir, classes)

    # Separă și copiază imaginile în directorul `test`
    for class_name, images in allocation.items():
        random.shuffle(images)
        test_split = int(len(images) * test_ratio)

        test_images = images[:test_split]

        # Copierea imaginilor în `test`, evitând duplicatele
        test_dir = output_dir / "test" / source / class_name
        copy_images_with_no_duplicates(test_images, test_dir)


def main():
    while True:
        # Introducere date din consolă
        source_dir = input("Enter the source directory for images (default: ./data_project/preprocessed/Custom): ").strip()
        if not source_dir:
            source_dir = "./data_project/preprocessed/Custom"

        output_directory = input("Enter the output directory for split datasets (default: ./data_project/dataset_split): ").strip()
        if not output_directory:
            output_directory = "./data_project/dataset_split"

        source_name = input("Enter the name of the dataset (default: Custom): ").strip()
        if not source_name:
            source_name = "Custom"

        class_names = input("Enter class names separated by commas (e.g., Oameni, Caini, Pisici, Masini, Cai, Cladiri): ").strip()
        if not class_names:
            raise ValueError("No class names provided. Please enter at least one class.")
        classes = [name.strip() for name in class_names.split(",")]

        # Procesarea dataset-ului pentru "test" doar
        try:
            print(f"Processing dataset for source: {source_name}")
            split_dataset_test_only(source_dir, output_directory, source_name, classes)
            print(f"Dataset successfully processed for source: {source_name}")
        except Exception as e:
            print(f"Error: {e}")

        # Întrebăm utilizatorul dacă dorește să proceseze alt set de date
        continue_choice = input("Do you want to process another dataset? (yes/no): ").strip().lower()
        if continue_choice != "yes":
            print("Exiting...")
            break


if __name__ == "__main__":
    main()
