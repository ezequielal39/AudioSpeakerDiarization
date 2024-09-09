import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return y, sr, mfcc

def clustering_kmeans(mfcc):
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
    labels = kmeans.fit_predict(mfcc.T)
    return labels

def aplicar_gmm(mfcc):
    gmm = GaussianMixture(n_components=2, covariance_type='diag')
    gmm.fit(mfcc.T)
    labels = gmm.predict(mfcc.T)
    return labels

def aplicar_hmm(mfcc):
    model = hmm.GaussianHMM(n_components=2, covariance_type='diag')
    model.fit(mfcc.T)
    states = model.predict(mfcc.T)
    return states

def diarizacion(audio_path):
    try:
        y, sr, mfcc = extraer_caracteristicas(audio_path)
        
        labels_kmeans = clustering_kmeans(mfcc)
        labels_gmm = aplicar_gmm(mfcc)
        states_hmm = aplicar_hmm(mfcc)

        tiempo = np.arange(len(labels_kmeans)) * (len(y) / len(labels_kmeans)) / sr

        plt.figure(figsize=(15, 10))

        plt.subplot(3, 1, 1)
        plt.plot(tiempo, labels_kmeans, label='K-means')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Etiqueta')
        plt.title('Resultados de K-means')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(tiempo, labels_gmm, label='GMM')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Etiqueta')
        plt.title('Resultados de GMM')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(tiempo, states_hmm, label='HMM')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Estado')
        plt.title('Resultados de HMM')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        messagebox.showerror("Error", str(e))

def seleccionar_archivo(entrada_archivo):
    archivo = filedialog.askopenfilename(
        title="Seleccionar archivo de audio",
        filetypes=(("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*"))
    )
    if archivo:
        entrada_archivo.set(archivo)

def crear_interfaz():
    ventana = tk.Tk()
    ventana.title("Diarización de Audio")

    etiqueta_archivo = tk.Label(ventana, text="Archivo de audio:")
    etiqueta_archivo.grid(row=0, column=0, padx=10, pady=10)
    
    entrada_archivo = tk.StringVar()
    campo_archivo = tk.Entry(ventana, textvariable=entrada_archivo, width=40)
    campo_archivo.grid(row=0, column=1, padx=10, pady=10)
    
    boton_archivo = tk.Button(ventana, text="Abrir", command=lambda: seleccionar_archivo(entrada_archivo))
    boton_archivo.grid(row=0, column=2, padx=10, pady=10)
    
    boton_ejecutar = tk.Button(ventana, text="Ejecutar", command=lambda: diarizacion(entrada_archivo.get()))
    boton_ejecutar.grid(row=1, column=1, padx=10, pady=10)
    
    ventana.mainloop()

crear_interfaz()
