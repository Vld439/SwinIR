"""
Filtros clásicos de restauración (módulo reutilizado desde la PoC del informe)

PoC: Mejora y restauración de imágenes
Procesamiento digital de imágenes - UNE, Facultad Politécnica

Compara técnicas de restauración en el dominio espacial y de la frecuencia
bajo tres escenarios de degradación, usando PSNR, SSIM y tiempo de cómputo.

Escenarios:
  E1  Ruido impulsivo (sal y pimienta, 5 %)
  E2  Ruido gaussiano aditivo (sigma = 0.08)
  E3  Desenfoque gaussiano (PSF) + ruido gaussiano (sigma = 0.01)

Técnicas:
  Espacial:    mediana, mediana adaptativa, Wiener adaptativo local,
               Non-Local Means
  Frecuencia:  filtro inverso, filtro de Wiener, mínimos cuadrados
               restringidos (CLS), media geométrica
Uso:
  python restauracion.py [--salida carpeta] [--semilla N]
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_nl_means


# --------------------------------------------------------------------------
# Modelo de degradación: g = h * f + n
# --------------------------------------------------------------------------
def psf_gaussiana(tam=15, sigma=2.0):
    ax = np.arange(tam) - tam // 2
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()


def psf_a_otf(h, forma):
    """Función de transferencia óptica con el centro de la PSF en el origen."""
    pad = np.zeros(forma)
    pad[: h.shape[0], : h.shape[1]] = h
    pad = np.roll(pad, (-(h.shape[0] // 2), -(h.shape[1] // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def ruido_sal_pimienta(f, prob, rng):
    g = f.copy()
    r = rng.random(f.shape)
    g[r < prob / 2] = 0.0
    g[(r >= prob / 2) & (r < prob)] = 1.0
    return g


def ruido_gaussiano(f, sigma, rng):
    return np.clip(f + rng.normal(0, sigma, f.shape), 0, 1)


def desenfoque(f, h):
    H = psf_a_otf(h, f.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(f) * H)), H


# --------------------------------------------------------------------------
# Dominio espacial
# --------------------------------------------------------------------------
def filtro_mediana(g, tam=3):
    return ndimage.median_filter(g, size=tam, mode="reflect")


def filtro_mediana_adaptativa(g, smax=7):
    """Mediana adaptativa (niveles A y B), vectorizada por tamaño de ventana."""
    salida = g.copy()
    pendiente = np.ones(g.shape, dtype=bool)
    for s in range(3, smax + 1, 2):
        zmin = ndimage.minimum_filter(g, size=s, mode="reflect")
        zmax = ndimage.maximum_filter(g, size=s, mode="reflect")
        zmed = ndimage.median_filter(g, size=s, mode="reflect")
        a_ok = (zmed - zmin > 0) & (zmed - zmax < 0)
        b_ok = (g - zmin > 0) & (g - zmax < 0)
        # Nivel A superado: decide el nivel B
        resuelto = pendiente & a_ok
        salida[resuelto & b_ok] = g[resuelto & b_ok]
        salida[resuelto & ~b_ok] = zmed[resuelto & ~b_ok]
        pendiente &= ~a_ok
        if s == smax:  # ventana máxima alcanzada
            salida[pendiente] = zmed[pendiente]
    return salida


def filtro_wiener_local(g, tam=5):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(signal.wiener(g, mysize=tam), 0, 1)


def estimar_sigma(g):
    """Estimador de desviación del ruido de Immerkaer (1996), sin dependencias extra."""
    mascara = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)
    alto, ancho = g.shape
    conv = signal.convolve2d(g, mascara, mode="valid")
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum()
                 / (6.0 * (ancho - 2) * (alto - 2)))


def filtro_nlm(g, h_factor=0.8):
    sigma = estimar_sigma(g)
    return np.clip(
        denoise_nl_means(
            g, h=h_factor * sigma, sigma=sigma, patch_size=5,
            patch_distance=6, fast_mode=True,
        ), 0, 1)


# --------------------------------------------------------------------------
# Dominio de la frecuencia (requieren conocer H)
# --------------------------------------------------------------------------
def filtro_inverso(g, H, radio=None):
    """Filtro inverso; con `radio` se limita la banda (inverso truncado)."""
    G = np.fft.fft2(g)
    Hs = np.where(np.abs(H) < 1e-8, 1e-8, H)
    F = G / Hs
    if radio is not None:
        u = np.fft.fftfreq(g.shape[0])[:, None]
        v = np.fft.fftfreq(g.shape[1])[None, :]
        F = F * (np.sqrt(u ** 2 + v ** 2) <= radio)
    return np.clip(np.real(np.fft.ifft2(F)), 0, 1)


def filtro_wiener_freq(g, H, K):
    """Wiener paramétrico: K aproxima el cociente S_n / S_f."""
    G = np.fft.fft2(g)
    W = np.conj(H) / (np.abs(H) ** 2 + K)
    return np.clip(np.real(np.fft.ifft2(W * G)), 0, 1)


def filtro_cls(g, H, gamma):
    """Mínimos cuadrados restringidos con Laplaciano como operador de suavidad."""
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)
    P = np.fft.fft2(lap, s=g.shape)
    G = np.fft.fft2(g)
    W = np.conj(H) / (np.abs(H) ** 2 + gamma * np.abs(P) ** 2)
    return np.clip(np.real(np.fft.ifft2(W * G)), 0, 1)


def filtro_media_geometrica(g, H, K, alfa=0.5):
    """F = [H*/|H|^2]^a [H*/(|H|^2 + K)]^(1-a) G.  a=1: inverso; a=0: Wiener."""
    G = np.fft.fft2(g)
    Hs2 = np.abs(H) ** 2 + 1e-12
    inv = np.conj(H) / Hs2
    wie = np.conj(H) / (Hs2 + K)
    # Ambos factores son reales positivos escalados por la fase de H*;
    # la potencia fraccionaria se aplica a la magnitud.
    mag = (np.abs(inv) ** alfa) * (np.abs(wie) ** (1 - alfa))
    W = mag * np.exp(-1j * np.angle(H))
    return np.clip(np.real(np.fft.ifft2(W * G)), 0, 1)


# --------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------
def evaluar(f, r):
    return psnr(f, r, data_range=1.0), ssim(f, r, data_range=1.0)


def mejor_param(fn, grilla, f):
    """Devuelve el parámetro de la grilla que maximiza el PSNR contra f."""
    return max(grilla, key=lambda p: psnr(f, fn(p), data_range=1.0))


def ejecutar(nombre, fn, f, g):
    t0 = time.perf_counter()
    r = fn(g)
    dt = time.perf_counter() - t0
    p, s = evaluar(f, r)
    return {"tecnica": nombre, "psnr": p, "ssim": s, "tiempo_s": dt}, r


def figura(titulo, f, g, resultados, ruta):
    imgs = [("Original", f), ("Degradada", g)] + [
        (f"{m['tecnica']}\nPSNR {m['psnr']:.2f} dB | SSIM {m['ssim']:.3f}", r)
        for m, r in resultados
    ]
    n = len(imgs)
    cols = 4
    filas = int(np.ceil(n / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(4 * cols, 4.2 * filas))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, (t, im) in zip(np.ravel(axs), imgs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout(h_pad=3.0)
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)
    rng = np.random.default_rng(args.semilla)

    imagenes = {
        "camera": img_as_float(data.camera()),
        "moon": img_as_float(data.moon()),
    }
    h = psf_gaussiana(15, 2.0)
    filas_csv = []

    for nom_img, f in imagenes.items():
        f = f[:512, :512]
        H = psf_a_otf(h, f.shape)

        # ---- E1: sal y pimienta ----
        g1 = ruido_sal_pimienta(f, 0.05, rng)
        tecs1 = [
            ("Sin restaurar", lambda g: g),
            ("Mediana 3x3", filtro_mediana),
            ("Mediana adaptativa", filtro_mediana_adaptativa),
            ("Wiener local 5x5", filtro_wiener_local),
            ("Non-Local Means", filtro_nlm),
        ]
        # ---- E2: gaussiano ----
        g2 = ruido_gaussiano(f, 0.08, rng)
        tecs2 = [
            ("Sin restaurar", lambda g: g),
            ("Mediana 3x3", filtro_mediana),
            ("Wiener local 5x5", filtro_wiener_local),
            ("Non-Local Means", filtro_nlm),
        ]
        # ---- E3: desenfoque + ruido ----
        gb, _ = desenfoque(f, h)
        g3 = np.clip(gb + rng.normal(0, 0.01, f.shape), 0, 1)
        K_teo = 0.01 ** 2 / np.var(f)  # S_n / S_f con espectro plano
        # Parámetros óptimos por búsqueda en grilla (oráculo: usa la original,
        # por lo que representa el mejor caso de cada método).
        r_opt = mejor_param(
            lambda p: filtro_inverso(g3, H, radio=p),
            np.linspace(0.04, 0.30, 27), f)
        K_opt = mejor_param(
            lambda p: filtro_wiener_freq(g3, H, p),
            np.logspace(-5, -0.5, 28), f)
        gam_opt = mejor_param(
            lambda p: filtro_cls(g3, H, p), np.logspace(-6, -1, 26), f)
        (a_opt, Kg_opt) = mejor_param(
            lambda p: filtro_media_geometrica(g3, H, p[1], p[0]),
            [(a, k) for a in (0.1, 0.3, 0.5, 0.7, 0.9)
             for k in np.logspace(-5, -0.5, 16)], f)
        tecs3 = [
            ("Sin restaurar", lambda g: g),
            ("Inverso", lambda g: filtro_inverso(g, H)),
            ("Inverso truncado (r=%.2f)" % r_opt,
             lambda g: filtro_inverso(g, H, radio=r_opt)),
            ("Wiener K teórico (%.1e)" % K_teo,
             lambda g: filtro_wiener_freq(g, H, K_teo)),
            ("Wiener K óptimo (%.1e)" % K_opt,
             lambda g: filtro_wiener_freq(g, H, K_opt)),
            ("CLS (gamma=%.1e)" % gam_opt,
             lambda g: filtro_cls(g, H, gam_opt)),
            ("Media geométrica (a=%.1f, K=%.0e)" % (a_opt, Kg_opt),
             lambda g: filtro_media_geometrica(g, H, Kg_opt, a_opt)),
        ]

        for esc, titulo, g, tecs in [
            ("E1", "E1 Sal y pimienta 5 %", g1, tecs1),
            ("E2", "E2 Ruido gaussiano sigma=0.08", g2, tecs2),
            ("E3", "E3 Desenfoque + ruido sigma=0.01", g3, tecs3),
        ]:
            res = []
            for nombre, fn in tecs:
                m, r = ejecutar(nombre, fn, f, g)
                res.append((m, r))
                filas_csv.append({"imagen": nom_img, "escenario": esc, **m})
            figura(f"{titulo} - imagen '{nom_img}'", f, g, res,
                   os.path.join(args.salida, f"{nom_img}_{esc}.png"))

    with open(os.path.join(args.salida, "metricas.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["imagen", "escenario", "tecnica", "psnr",
                            "ssim", "tiempo_s"])
        w.writeheader()
        for fila in filas_csv:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in fila.items()})
    print(f"Listo. Resultados en '{args.salida}/'")


if __name__ == "__main__":
    main()
