import cv2


def display_masked_image(image_path, mask_path):
    original_image = cv2.imread(image_path)
    mask_image = cv2.imread(mask_path)
    masked_image = cv2.addWeighted(original_image, 0.7, mask_image, 0.3, 0)
    cv2.imshow("Masked Image", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    filename = '1.1164.jpg'
    # filename = '1.1265.jpg'
    # filename = '10.1671.jpg'
    # filename = '11.2028.jpg'
    display_masked_image(f'../wsi-segment/image-tiles/{filename}',
                         f'../wsi-segment/mask-tiles/{filename}')