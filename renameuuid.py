import os
import uuid

def renombrar_con_uuid_personalizado(ruta_carpeta, longitud=13):
    if not os.path.exists(ruta_carpeta):
        print("La ruta no existe.")
        return

    for nombre_archivo in os.listdir(ruta_carpeta):
        ruta_antigua = os.path.join(ruta_carpeta, nombre_archivo)

        if os.path.isfile(ruta_antigua):
            nombre_base, extension = os.path.splitext(nombre_archivo)
            id_unico = str(uuid.uuid4())[:longitud]
            
            nuevo_nombre = f"{id_unico}{extension}"
            ruta_nueva = os.path.join(ruta_carpeta, nuevo_nombre)

            try:
                os.rename(ruta_antigua, ruta_nueva)
                print(f"Ok: {nombre_archivo} -> {nuevo_nombre}")
            except Exception as e:
                print(f"Error en {nombre_archivo}: {e}")

if __name__ == "__main__":
    mi_ruta = '/home/jasa/Documents/PersonsReady/Models'
    largo_deseado = 8

    renombrar_con_uuid_personalizado(mi_ruta, largo_deseado)