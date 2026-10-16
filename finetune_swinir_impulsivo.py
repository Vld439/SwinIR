"""
Aporte propio: fine-tuning de SwinIR (denoising en grises) para ruido impulsivo
===============================================================================
SwinIR solo fue entrenado con ruido gaussiano, por eso falla frente a ruido de
sal y pimienta. Este script parte de los pesos oficiales (noise25) y los reajusta
durante unos miles de pasos con imágenes degradadas al vuelo:

  - ruido de sal y pimienta con probabilidad p ~ U(0.01, 0.20), y
  - una fracción de ejemplos con ruido gaussiano (sigma ~ U(10, 40)/255) para que
    el modelo no olvide lo que ya sabía (parámetro --mezcla_gauss).

Los pesos resultantes se guardan en el mismo formato que los oficiales
({'params': ...}), de modo que main_test_swinir.py y comparar_swinir_clasicos.py
los cargan con --model_path / --pesos-ft sin cambios.

Uso (desde la raíz del repositorio SwinIR, con GPU):
  python finetune_swinir_impulsivo.py --datos RUTA/A/CARPETA_DE_FOTOS --iters 20   # prueba rápida
  python finetune_swinir_impulsivo.py --datos RUTA/A/CARPETA_DE_FOTOS               # entrenamiento

Importante: --datos NO debe contener las imágenes de testsets/Set12, que se usan
para evaluar.
"""
import argparse
import os
import random
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from main_test_swinir import define_model  # código original del repositorio

VENTANA = 8
EXTENSIONES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
PESOS_OFICIALES = "model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth"


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
def listar_imagenes(carpeta):
    rutas = []
    for raiz, _, archivos in os.walk(carpeta):
        for a in archivos:
            if a.lower().endswith(EXTENSIONES):
                rutas.append(os.path.join(raiz, a))
    return sorted(rutas)


def cargar_entrenamiento(carpeta, lado_minimo):
    """Carga en memoria (uint8, gris) las imágenes con lado >= lado_minimo."""
    imgs = []
    for r in listar_imagenes(carpeta):
        im = cv2.imread(r, cv2.IMREAD_GRAYSCALE)
        if im is not None and min(im.shape) >= lado_minimo:
            imgs.append(im)
    return imgs


def cargar_validacion(carpeta):
    imgs = []
    for r in listar_imagenes(carpeta):
        im = cv2.imread(r, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            imgs.append(im.astype(np.float32) / 255.0)
    return imgs


def lote(imgs, batch, parche, dispositivo):
    """Recortes aleatorios de parche x parche con volteos y rotaciones."""
    xs = []
    for _ in range(batch):
        im = random.choice(imgs)
        h, w = im.shape
        y = random.randint(0, h - parche)
        x = random.randint(0, w - parche)
        p = im[y:y + parche, x:x + parche]
        if random.random() < 0.5:
            p = p[:, ::-1]
        if random.random() < 0.5:
            p = p[::-1, :]
        p = np.rot90(p, random.randint(0, 3))
        xs.append(np.ascontiguousarray(p, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(xs))[:, None].to(dispositivo)


def degradar(limpia, frac_gauss):
    """Ruido impulsivo (o, con prob. frac_gauss, gaussiano) sobre un lote (B,1,H,W)."""
    b = limpia.shape[0]
    dev = limpia.device
    p = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.01, 0.20)
    r = torch.rand_like(limpia)
    impulsiva = limpia.clone()
    impulsiva[r < p / 2] = 0.0
    impulsiva[(r >= p / 2) & (r < p)] = 1.0
    sigma = torch.empty(b, 1, 1, 1, device=dev).uniform_(10, 40) / 255.0
    gaussiana = limpia + torch.randn_like(limpia) * sigma
    usar_gauss = torch.rand(b, 1, 1, 1, device=dev) < frac_gauss
    return torch.where(usar_gauss, gaussiana, impulsiva)


def charbonnier(a, b, eps=1e-3):
    """Variante suave de la pérdida L1."""
    return torch.sqrt((a - b) ** 2 + eps ** 2).mean()


# --------------------------------------------------------------------------
# Evaluación (imagen completa, con el mismo relleno que main_test_swinir.py)
# --------------------------------------------------------------------------
@torch.no_grad()
def restaurar(modelo, img, dispositivo):
    x = torch.from_numpy(img.astype(np.float32))[None, None].to(dispositivo)
    _, _, h, w = x.shape
    hp = (h // VENTANA + 1) * VENTANA - h
    wp = (w // VENTANA + 1) * VENTANA - w
    x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h + hp, :]
    x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w + wp]
    y = modelo(x)
    return y[..., :h, :w].squeeze().float().cpu().clamp_(0, 1).numpy()


def psnr_db(y, f):
    return 10 * np.log10(1.0 / max(float(np.mean((y - f) ** 2)), 1e-12))


def evaluar(modelo, val_imgs, dispositivo):
    """PSNR medio en Set12 con sal y pimienta (5 %) y con ruido gaussiano (25/255)."""
    modelo.eval()
    rng = np.random.default_rng(0)
    imp, gau = [], []
    for f in val_imgs:
        r = rng.random(f.shape)
        g_imp = f.copy()
        g_imp[r < 0.025] = 0.0
        g_imp[(r >= 0.025) & (r < 0.05)] = 1.0
        g_gau = f + rng.normal(0, 25 / 255.0, f.shape).astype(np.float32)
        imp.append(psnr_db(restaurar(modelo, g_imp, dispositivo), f))
        gau.append(psnr_db(restaurar(modelo, g_gau, dispositivo), f))
    modelo.train()
    return float(np.mean(imp)), float(np.mean(gau))


def guardar(modelo, ruta):
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    # mismo formato que los pesos oficiales: define_model lee la clave 'params'
    torch.save({"params": modelo.state_dict()}, ruta)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", required=True, help="carpeta con fotos de entrenamiento")
    ap.add_argument("--val", default="testsets/Set12", help="carpeta de validación")
    ap.add_argument("--pesos_iniciales", default=PESOS_OFICIALES)
    ap.add_argument("--salida", default="model_zoo/swinir/gray_dn_impulsivo_ft.pth",
                    help="pesos con mejor promedio en validación")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=4, help="bájelo si se queda sin memoria de GPU")
    ap.add_argument("--parche", type=int, default=128, help="múltiplo de 8")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--mezcla_gauss", type=float, default=0.25,
                    help="fracción de ejemplos con ruido gaussiano (evita olvidar)")
    ap.add_argument("--eval_cada", type=int, default=500)
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    assert args.parche % VENTANA == 0, "--parche debe ser múltiplo de 8"
    if not os.path.exists(args.pesos_iniciales):
        raise SystemExit(f"Faltan los pesos iniciales: {args.pesos_iniciales}\n"
                         "Corra primero el test original (paso 3 de la guía), que los descarga.")
    carpeta_datos = os.path.abspath(args.datos)
    carpeta_val = os.path.abspath(args.val)
    if carpeta_val == carpeta_datos or carpeta_val.startswith(carpeta_datos + os.sep):
        raise SystemExit("La carpeta de validación está dentro de --datos: se mezclaría "
                         "entrenamiento y evaluación. Use otra carpeta de fotos.")

    random.seed(args.semilla)
    np.random.seed(args.semilla)
    torch.manual_seed(args.semilla)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dev}")
    if dev.type == "cpu":
        print("AVISO: sin GPU el entrenamiento será extremadamente lento.")

    imgs = cargar_entrenamiento(args.datos, args.parche)
    val = cargar_validacion(args.val)
    assert imgs, f"No se encontraron imágenes de al menos {args.parche} px en {args.datos}"
    assert val, f"No hay imágenes de validación en {args.val}"
    print(f"Entrenamiento: {len(imgs)} imágenes | Validación: {len(val)} imágenes")

    a = SimpleNamespace(task="gray_dn", model_path=args.pesos_iniciales, scale=1,
                        noise=25, training_patch_size=128, large_model=False)
    modelo = define_model(a).to(dev)
    modelo.train()
    opt = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-6)

    base_imp, base_gau = evaluar(modelo, val, dev)
    print(f"ANTES  | sal y pimienta: {base_imp:.2f} dB | gaussiano: {base_gau:.2f} dB")

    mejor = -1e9
    t0 = time.time()
    for it in range(1, args.iters + 1):
        limpia = lote(imgs, args.batch, args.parche, dev)
        ruidosa = degradar(limpia, args.mezcla_gauss)
        perdida = charbonnier(modelo(ruidosa), limpia)
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        sched.step()

        if it == 1 or it % 50 == 0:
            seg = time.time() - t0
            eta = seg / it * (args.iters - it)
            print(f"it {it:6d}/{args.iters} | pérdida {perdida.item():.5f} | "
                  f"{it / seg:.2f} it/s | faltan ~{eta / 60:.1f} min")
        if it % args.eval_cada == 0 or it == args.iters:
            imp, gau = evaluar(modelo, val, dev)
            puntaje = (imp + gau) / 2
            marca = ""
            if puntaje > mejor:
                mejor = puntaje
                guardar(modelo, args.salida)
                marca = "  <- guardado (mejor promedio)"
            print(f"  VAL it {it}: sal y pimienta {imp:.2f} dB | gaussiano {gau:.2f} dB{marca}")

    ultimo = os.path.splitext(args.salida)[0] + "_ultimo.pth"
    guardar(modelo, ultimo)
    imp, gau = evaluar(modelo, val, dev)
    print("\nResumen (PSNR medio en Set12)")
    print(f"  Sal y pimienta 5 %: {base_imp:.2f} -> {imp:.2f} dB")
    print(f"  Gaussiano 25/255 : {base_gau:.2f} -> {gau:.2f} dB")
    print(f"Pesos guardados en: {args.salida} y {ultimo}")
    print(f"Siguiente paso: python comparar_swinir_clasicos.py --pesos-ft {args.salida}")


if __name__ == "__main__":
    main()
