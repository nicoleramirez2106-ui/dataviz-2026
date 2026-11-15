"""Diagnóstico de la conexión con Gemini. Correr desde la carpeta del repositorio:

    python dashboard/probar_gemini.py

No muestra la clave: solo dice si existe, qué modelos puede usar y qué responde Google.
"""
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ia  # noqa: E402

print("Python", sys.version.split()[0], "·", ssl.OPENSSL_VERSION)
clave = os.environ.get("GEMINI_API_KEY", "")
print("1) Clave GEMINI_API_KEY:", f"encontrada ({len(clave)} caracteres)" if clave else "NO encontrada")
if not clave:
    sys.exit("   Configure la clave con setx y vuelva a abrir VS Code.")
if clave != clave.strip() or clave.startswith(('"', "'")):
    print("   Ojo: la clave tiene espacios o comillas de más al principio o al final.")

print("2) Modelos que su clave puede usar:")
try:
    lista = ia.disponibles(clave)
    print("   " + (", ".join(lista[:12]) if lista else "ninguno 'flash'"))
except Exception as err:
    print("   No se pudo consultar:", ia._detalle(err))
    lista = []

print("3) Prueba corta con cada modelo:")
for modelo in (ia.MODELOS + [m for m in lista if m not in ia.MODELOS])[:8]:
    try:
        r = ia._llamar(modelo, "Responda solo: listo", clave)
        print(f"   {modelo}: OK -> {r[:40]!r}")
        break
    except Exception as err:
        print(f"   {modelo}: {ia._detalle(err)}")
