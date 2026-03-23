import argparse
import os
import string
import uuid


def renombrar_con_uuid_personalizado(ruta_carpeta, longitud=13, shuffle=False):
    if not os.path.exists(ruta_carpeta):
        print("La ruta no existe.")
        return

    hex_digits = set(string.hexdigits)

    for nombre_archivo in os.listdir(ruta_carpeta):
        ruta_antigua = os.path.join(ruta_carpeta, nombre_archivo)

        if os.path.isfile(ruta_antigua):
            nombre_base, extension = os.path.splitext(nombre_archivo)

            if (
                not shuffle
                and len(nombre_base) == longitud
                and set(nombre_base) <= hex_digits
            ):
                continue

            id_unico = str(uuid.uuid4())[:longitud]
            nuevo_nombre = f"{id_unico}{extension}"
            ruta_nueva = os.path.join(ruta_carpeta, nuevo_nombre)

            try:
                os.rename(ruta_antigua, ruta_nueva)
                print(f"OK: {nombre_archivo} -> {nuevo_nombre}")
            except Exception as e:
                print(f"Error en {nombre_archivo}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--path", default="/home/jasa/Documents/PersonsReady/Models")
    parser.add_argument("--length", type=int, default=8)
    args = parser.parse_args()

    renombrar_con_uuid_personalizado(args.path, args.length, args.shuffle)
