import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

user_id = input('Masukan Nama Pengguna: ')
dataset_path = f'dataset/{user_id}'

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    print(f"direktori penyimpanan '{dataset_path}' berhasil dibuat")

cap = cv2.VideoCapture(0)
count = 0

print("\n--- Instruksi ---")
print("Arahkan wajah ke kamera")
print("Tekan tombol 's' pada keyboard untuk mengambil dan menyimpan sampel gambar")
print("Tekan tombol 'q' untuk keluar jika sudah cukup (target 30-50 sampel)")

while True:
    ret, frame = cap.read()
    if not ret:
        print('Gagal membaca frame dari webcam')
        break

    # tahapan image preproc
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # untuk konversi ke grayscale
    filtered = cv2.GaussianBlur(gray, (5, 5), 0)     # reduksi noise
    enhanced = cv2.equalizeHist(filtered)            # untuk mengatasi variasi pencahayaan ruangan
    faces = face_cascade.detectMultiScale(enhanced, scaleFactor=1.3, minNeighbors=5)

    clean_face = None
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = enhanced[y:y+h, x:x+w]
        clean_face = cv2.resize(face_roi, (100, 100))
        break

    cv2.putText(frame, f"Samples saved: {count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('raw data collection & preprocessing', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if clean_face is not None:
            count += 1
            file_name = f"{dataset_path}/img_{count}.jpg"
            cv2.imwrite(file_name, clean_face)
            print(f'[Berhasil Disimpan] {file_name}')
        else:
            print('[Peringatan] Wajah belum terdeteksi, posisikan wajah menghadap kamera.')
    elif key == ord('q') or count >= 40:
        break

cap.release()
cv2.destroyAllWindows()
print(f'\nPengambilan data selesai, total {count} clean dataset berhasil di simpan di dalam folder {dataset_path}')