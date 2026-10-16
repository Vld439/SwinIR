"""
Contribución propia sobre SwinIR (Liang et al., 2021)
=====================================================
Compara el modelo preentrenado de denoising en escala de grises de SwinIR
con filtros clásicos de restauración (mediana, mediana adaptativa, Wiener
local, Non-Local Means, inverso truncado, Wiener, CLS y media geométrica)
en tres escenarios de degradación, sobre las 12 imágenes de testsets/Set12.

Escenarios
  E1  Ruido de sal y pimienta (5 %)
  E2  Ruido gaussiano, sigma = 25/255 (el nivel con que se entrenó el modelo)
  E3  Desenfoque gaussiano (PSF 15x15, sigma 2) + ruido gaussiano sigma 0.01

Aporte: SwinIR solo fue entrenado con ruido gaussiano. E1 y E3 miden qué pasa
cuando la degradación está fuera de su distribución de entrenamiento.

Uso (desde la raíz del repositorio SwinIR):
  python comparar_swinir_clasicos.py                 # con SwinIR (GPU si hay)
  python comparar_swinir_clasicos.py --sin-swinir    # solo filtros clásicos
"""
import argparse
import csv
import glob
import os
import time
from collections import defaultdict
from types import SimpleNamespace

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

from restauracion_clasica import (
    desenfoque, estimar_sigma, filtro_cls, filtro_inverso,
    filtro_media_geometrica, filtro_mediana, filtro_mediana_adaptativa,
    filtro_nlm, filtro_wiener_freq, filtro_wiener_local, mejor_param,
    psf_a_otf, psf_gaussiana, ruido_gaussiano, ruido_sal_pimienta,
)
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

URL_PESOS = ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
             "{}")
PESOS = "model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth"
VENTANA = 8  # window_size de SwinIR para denoising


# --------------------------------------------------------------------------
# SwinIR (reutiliza define_model del repositorio)
# --------------------------------------------------------------------------
class SwinIRGris:
    def __init__(self, ruta_pesos=PESOS):
        import torch
        from main_test_swinir import define_model  # código original del repo
        self.torch = torch
        if not os.path.exists(ruta_pesos) and ruta_pesos != PESOS:
            raise SystemExit(f"No existe el archivo de pesos: {ruta_pesos}")
        if not os.path.exists(ruta_pesos):
            os.makedirs(os.path.dirname(ruta_pesos), exist_ok=True)
            url = URL_PESOS.format(os.path.basename(ruta_pesos))
            print(f"Descargando pesos: {url}")
            r = requests.get(url, allow_redirects=True, timeout=300)
            r.raise_for_status()
            with open(ruta_pesos, "wb") as fh:
                fh.write(r.content)
        args = SimpleNamespace(task="gray_dn", model_path=ruta_pesos,
                               scale=1, noise=25, training_patch_size=128,
                               large_model=False)
        self.dispositivo = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = define_model(args).eval().to(self.dispositivo)
        print(f"SwinIR cargado en: {self.dispositivo}")

    def __call__(self, img):
        """img: float32 (H, W) en [0, 1] -> restaurada (H, W) en [0, 1]."""
        t = self.torch
        x = t.from_numpy(img.astype(np.float32))[None, None].to(self.dispositivo)
        _, _, h, w = x.shape
        hp = (h // VENTANA + 1) * VENTANA - h
        wp = (w // VENTANA + 1) * VENTANA - w
        # mismo relleno por reflexión que main_test_swinir.py
        x = t.cat([x, t.flip(x, [2])], 2)[:, :, :h + hp, :]
        x = t.cat([x, t.flip(x, [3])], 3)[:, :, :, :w + wp]
        with t.no_grad():
            y = self.modelo(x)
        y = y[..., :h, :w].squeeze().float().cpu().clamp_(0, 1).numpy()
        return y

    def sincronizar(self):
        if self.dispositivo.type == "cuda":
            self.torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Escenarios
# --------------------------------------------------------------------------
def con_modelos(t, g, modelos):
    """Agrega a la lista t una entrada por cada variante de SwinIR."""
    return t + [(nombre, (lambda m=m: m(g))) for nombre, m in modelos.items()]


def tecnicas_E1(g, f, modelos):
    t = [("Mediana 3x3", lambda: filtro_mediana(g)),
         ("Mediana adaptativa", lambda: filtro_mediana_adaptativa(g)),
         ("Wiener local 5x5", lambda: filtro_wiener_local(g)),
         ("Non-Local Means", lambda: filtro_nlm(g))]
    return con_modelos(t, g, modelos)


def tecnicas_E2(g, f, modelos):
    t = [("Mediana 3x3", lambda: filtro_mediana(g)),
         ("Wiener local 5x5", lambda: filtro_wiener_local(g)),
         ("Non-Local Means", lambda: filtro_nlm(g))]
    return con_modelos(t, g, modelos)


def tecnicas_E3(g, f, modelos, H):
    # Parámetros por búsqueda en grilla con la imagen original (caso óptimo)
    r = mejor_param(lambda p: filtro_inverso(g, H, radio=p),
                    np.linspace(0.04, 0.30, 27), f)
    K = mejor_param(lambda p: filtro_wiener_freq(g, H, p),
                    np.logspace(-5, -0.5, 28), f)
    gam = mejor_param(lambda p: filtro_cls(g, H, p),
                      np.logspace(-6, -1, 26), f)
    a, Kg = mejor_param(
        lambda p: filtro_media_geometrica(g, H, p[1], p[0]),
        [(a, k) for a in (0.1, 0.3, 0.5, 0.7, 0.9)
         for k in np.logspace(-5, -0.5, 16)], f)
    t = [("Inverso truncado", lambda: filtro_inverso(g, H, radio=r)),
         ("Wiener (K optimo)", lambda: filtro_wiener_freq(g, H, K)),
         ("CLS", lambda: filtro_cls(g, H, gam)),
         ("Media geometrica", lambda: filtro_media_geometrica(g, H, Kg, a))]
    return con_modelos(t, g, modelos)


def medir(fn, swin):
    if swin:
        swin.sincronizar()
    t0 = time.perf_counter()
    r = fn()
    if swin:
        swin.sincronizar()
    return r, time.perf_counter() - t0


def figura(escenario, nombre_img, f, g, res, ruta):
    imgs = [("Original", f), ("Degradada", g)] + [
        (f"{n}\nPSNR {p:.2f} | SSIM {s:.3f}", r) for n, r, p, s in res]
    cols = 4
    filas = int(np.ceil(len(imgs) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(4 * cols, 4.3 * filas))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, (t, im) in zip(np.ravel(axs), imgs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
    fig.suptitle(f"{escenario} - {nombre_img}", fontsize=13)
    fig.tight_layout(h_pad=3.0)
    fig.savefig(ruta, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default="testsets/Set12")
    ap.add_argument("--salida", default="resultados_swinir")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--sin-swinir", action="store_true",
                    help="omite SwinIR (útil para probar sin PyTorch)")
    ap.add_argument("--pesos-ft", default=None,
                    help="pesos ajustados con finetune_swinir_impulsivo.py; "
                         "se comparan junto al SwinIR original")
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    modelos = {}
    if not args.sin_swinir:
        modelos["SwinIR"] = SwinIRGris()
        if args.pesos_ft:
            modelos["SwinIR fine-tuned"] = SwinIRGris(args.pesos_ft)
        for m in modelos.values():  # calentamiento: el primer tiempo no incluye la carga
            m(np.zeros((64, 64), dtype=np.float32))

    rutas = sorted(glob.glob(os.path.join(args.carpeta, "*")))
    assert rutas, f"No hay imágenes en {args.carpeta}"
    h = psf_gaussiana(15, 2.0)
    detalle = []

    for idx, ruta in enumerate(rutas):
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        f = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        rng = np.random.default_rng(args.semilla + idx)
        H = psf_a_otf(h, f.shape)

        g1 = ruido_sal_pimienta(f, 0.05, rng)
        g2 = ruido_gaussiano(f, 25 / 255.0, rng)
        gb, _ = desenfoque(f, h)
        g3 = np.clip(gb + rng.normal(0, 0.01, f.shape), 0, 1)

        for esc, g, tecs in [
            ("E1 sal y pimienta", g1, tecnicas_E1(g1, f, modelos)),
            ("E2 gaussiano", g2, tecnicas_E2(g2, f, modelos)),
            ("E3 desenfoque + ruido", g3, tecnicas_E3(g3, f, modelos, H)),
        ]:
            base = (psnr(f, g, data_range=1.0), ssim(f, g, data_range=1.0))
            detalle.append([nombre, esc, "Sin restaurar", *base, 0.0])
            res = []
            for tec, fn in tecs:
                r, dt = medir(fn, modelos.get(tec))
                r = r.astype(np.float64)
                p, s = psnr(f, r, data_range=1.0), ssim(f, r, data_range=1.0)
                detalle.append([nombre, esc, tec, p, s, dt])
                res.append((tec, r, p, s))
            if idx == 0:
                figura(esc, nombre, f, g, res,
                       os.path.join(args.salida,
                                    f"ejemplo_{esc.split()[0]}.png"))
        print(f"[{idx + 1}/{len(rutas)}] {nombre} listo")

    with open(os.path.join(args.salida, "detalle.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["imagen", "escenario", "tecnica", "psnr", "ssim",
                    "tiempo_s"])
        w.writerows([[a, b, c, round(d, 4), round(e, 4), round(t, 4)]
                     for a, b, c, d, e, t in detalle])

    acum = defaultdict(list)
    for _, esc, tec, p, s, t in detalle:
        acum[(esc, tec)].append((p, s, t))
    filas = []
    for (esc, tec), v in acum.items():
        m = np.mean(v, axis=0)
        sd = np.std([x[0] for x in v])
        filas.append([esc, tec, round(m[0], 2), round(sd, 2), round(m[1], 3),
                      round(m[2], 3)])
    with open(os.path.join(args.salida, "metricas_promedio.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["escenario", "tecnica", "psnr_medio", "psnr_desv",
                    "ssim_medio", "tiempo_medio_s"])
        w.writerows(filas)

    print(f"\nPromedio sobre {len(rutas)} imágenes")
    print(f"{'escenario':24}{'técnica':22}{'PSNR':>8}{'±':>7}{'SSIM':>8}"
          f"{'t (s)':>8}")
    for f_ in filas:
        print(f"{f_[0]:24}{f_[1]:22}{f_[2]:8.2f}{f_[3]:7.2f}{f_[4]:8.3f}"
              f"{f_[5]:8.3f}")


if __name__ == "__main__":
    main()
