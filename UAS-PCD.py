import cv2
import numpy as np

def detect_and_save_objects(image_path, output_folder='output'):
    # Membaca gambar
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Gambar tidak ditemukan. Pastikan path gambar sudah benar.")
    
    # Mengubah gambar ke skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Menggunakan Gaussian Blur untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Menggunakan Adaptive Thresholding untuk memisahkan objek dari latar belakang
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Menggunakan Morphological Transformations untuk membersihkan noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned_thresh = cv2.morphologyEx(cleaned_thresh, cv2.MORPH_OPEN, kernel)
    
    # Mencari kontur dalam gambar
    contours, _ = cv2.findContours(cleaned_thresh, 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Menyimpan objek yang terdeteksi sebagai file PNG dan menampilkan ukurannya
    for i, contour in enumerate(contours):
        # Membuat bounding box
        x, y, w, h = cv2.boundingRect(contour)
        object_image = image[y:y+h, x:x+w]
        
        # Menyimpan gambar objek sebagai PNG
        output_path = f"{output_folder}/object_{i+1}.png"
        cv2.imwrite(output_path, object_image)
        
        # Menghitung luas dan perimeter
        area = cv2.contourArea(contour)
        parameter = cv2.arcLength(contour, True)
        
        print(f"Object {i+1}:")
        print(f" - Saved at: {output_path}")
        print(f" - Area: {area} pixels")
        print(f" - Parameter: {parameter} pixels")
      
    # Menggambar kontur pada gambar asli
    hasil_image = image.copy()
    cv2.drawContours(hasil_image, contours, -1, (0, 255, 0), 2)
    
    # Menampilkan gambar dengan kontur
    cv2.imshow("Deteksi Objek", hasil_image)
    cv2.waitKey(0)  # Menunggu sampai jendela ditutup
    cv2.destroyAllWindows()
    
    # Menyimpan hasil gambar dengan kontur
    cv2.imwrite('Deteksi Objek.png', hasil_image)

# Path gambar yang ingin diproses
path_gambar = 'images/Tes.png'
output_folder = 'images/Hasil Objek Terdeteksi'  
# Folder tempat menyimpan objek yang terdeteksi

# Memanggil fungsi untuk mendeteksi dan menyimpan objek
detect_and_save_objects(path_gambar, output_folder)


