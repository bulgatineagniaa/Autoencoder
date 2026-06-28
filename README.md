# Autoencoder Fashion-MNIST

## Nama
**Bulgatine Agnia**

## Mata Kuliah
Machine Learning 2

---

# Deskripsi

Proyek ini merupakan implementasi **Autoencoder** menggunakan dataset **Fashion-MNIST** dengan framework **PyTorch**.

Autoencoder digunakan untuk mempelajari representasi (encoding) dari citra kemudian merekonstruksi kembali citra tersebut menggunakan Decoder.

Pada proyek ini dilakukan percobaan dengan beberapa ukuran latent space, yaitu:

- Latent Dimension = 2
- Latent Dimension = 8
- Latent Dimension = 32

Model dilatih menggunakan dataset Fashion-MNIST dan disimpan dalam format `.pth`.

---

# Dataset

Dataset yang digunakan:

**Fashion-MNIST**

Dataset akan otomatis diunduh melalui torchvision.

Jumlah data:

- Training : 60.000 gambar
- Testing : 10.000 gambar

Ukuran gambar:

```
28 x 28 grayscale
```

---

# Library

Project ini menggunakan:

- Python
- PyTorch
- Torchvision
- Matplotlib
- NumPy

Install dependency:

```bash
pip install torch torchvision matplotlib numpy
```

---

# Struktur Folder

```
results/

│── autoencoder_latent2.pth
│── autoencoder_latent8.pth
│── autoencoder_latent32.pth

│── encoder_fashion_mnist.pth
│── decoder_fashion_mnist.pth

│── reconstruct.py
│── generate_from_decoder.py

│── original.png
│── reconstructed.png
│── comparison.png
│── generated_image.png

│── README.md
```

---

# Menjalankan Rekonstruksi

Contoh:

```bash
python reconstruct.py --model autoencoder_latent32.pth --index 10
```

Output:

- original.png
- reconstructed.png
- comparison.png

---

# Menjalankan Decoder

Contoh:

```bash
python generate_from_decoder.py --decoder decoder_fashion_mnist.pth --latent "0.5,-1.2,0.3,0.8,0.2,-0.5,0.1,0.9,0.4,-0.6,0.3,0.8,-0.1,0.2,-0.3,0.4,0.5,-0.2,0.6,-0.4,0.7,-0.8,0.2,0.1,-0.7,0.5,0.3,-0.2,0.4,-0.1,0.6,-0.5"
```

Output:

```
generated_image.png
```

---

# Hasil

Model mampu:

- Melakukan encoding citra Fashion-MNIST.
- Melakukan decoding untuk menghasilkan kembali citra.
- Merekonstruksi gambar dengan kualitas yang cukup baik.
- Menghasilkan citra baru menggunakan Decoder dari latent vector.

---

# Author

Bulgatine Agnia

Machine Learning 2
