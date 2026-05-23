import os
import zipfile

def zip_project(output_filename="proyecto_codigo.zip"):
    # Directorios y archivos a incluir
    include_dirs = ["src", "reports"]
    include_files = ["requirements.txt"]
    
    # Extensiones de archivos que queremos incluir si están en la raíz
    include_extensions = [".ipynb", ".py", ".md", ".txt"]
    
    # Directorios a excluir explícitamente
    exclude_dirs = [".venv", "data", "models", ".git", "__pycache__", "scratch"]
    
    print(f"[~] Creando archivo comprimido '{output_filename}'...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Agregar archivos específicos de la raíz
        for file in include_files:
            if os.path.exists(file):
                zipf.write(file)
                print(f" [+] Agregado: {file}")
                
        # 2. Buscar archivos permitidos en la raíz
        for item in os.listdir("."):
            if os.path.isfile(item):
                ext = os.path.splitext(item)[1]
                if ext in include_extensions and item not in include_files and item != output_filename and item != "zip_project.py":
                    zipf.write(item)
                    print(f" [+] Agregado: {item}")
        
        # 3. Recorrer y agregar directorios incluidos
        for folder in include_dirs:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    # Filtrar carpetas excluidas como __pycache__ dentro de src
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path)
                        print(f" [+] Agregado: {file_path}")

    print(f"\n[+] ¡Listo! El proyecto se ha comprimido en '{output_filename}'.")
    print("    Puedes subir este archivo directamente a Google Drive o Kaggle.")

if __name__ == "__main__":
    zip_project()
