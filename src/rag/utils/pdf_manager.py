from pdf2image import convert_from_path
import os
import shutil

class PdfManager:
    def __init__(self):
        pass

    def clear_and_recreate_dir(self, output_folder: str) -> None:
        print(f"Clearing output folder {output_folder}")

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        os.makedirs(output_folder)

    def save_images(self, id: str, pdf_path: str) -> list[str]:
        output_folder = f"data/processed/{id}/"
        # Read pdf as images
        images = convert_from_path(pdf_path)
        # Create or clear and recreate folder
        # self.clear_and_recreate_dir(output_folder)
        prev_imgs = len(os.listdir(output_folder))

        # Start to save image
        num_of_imgs = len(images)
        num_of_processed_imgs = 0

        for i, image in enumerate(images):
            full_save_path = f"{output_folder}/page_{i+prev_imgs}.png"
            image.save(full_save_path, "PNG")
            num_of_processed_imgs += 1
        
        # Check for unprocessed image
        print(f"The number of unprocessed images is: {num_of_imgs - num_of_processed_imgs}")

        return [f"{output_folder}/page_{i+1}.png" for i in range(num_of_processed_imgs)]









