import os, sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def resize_image(source_image_path, output_image_path, output_size):
    """
    Resizes an image to the specified size.

    :param source_image_path: source image path
    :param output_image_path: destination image path
    :param output_size: The requested size in pixels, as a 2-tuple:(width, height).
    """
    try:
        with Image.open(source_image_path) as img:
            img.thumbnail(output_size, Image.NEAREST)
            img.save(output_image_path)
            print(f"Resized image saved to '{output_image_path}'")
    except IOError:
        print(f"Cannot resize_image for '{source_image_path}'")


def resize_images_in_folder(source_folder, output_folder, output_size):
    """

        :param source_folder: The folder containing the images to resize.
        :param output_folder: The folder to save the resized images.
        :param output_size: The requested size in pixels, as a 2-tuple:(width, height).
    """

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            input_path = os.path.join(source_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, output_size)


def main():
    if len(sys.argv) < 4:
        print("Usage: python image_resizer.py <source_folder> <output_folder> <output_size_x> <output_size_y>")
        sys.exit(1)

    source_folder = sys.argv[1]
    output_folder = sys.argv[2]
    size = int(sys.argv[3]), int(sys.argv[4])

    resize_images_in_folder(source_folder, output_folder, size)


if __name__ == '__main__':
    print(os.getcwd())
    source_folder = '../img/A'
    output_folder = '../img/B'
    output_size = (10000, 10000)
    resize_images_in_folder(source_folder, output_folder, output_size)
