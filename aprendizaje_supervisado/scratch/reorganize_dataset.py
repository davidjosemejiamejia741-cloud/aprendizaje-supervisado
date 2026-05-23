import os
import shutil

def reorganize():
    # Obtener la raíz del proyecto a partir de la ubicación de este script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    dataset_src = os.path.join(project_root, 'Dataset')
    data_dir = os.path.join(project_root, 'data')
    
    # 1. Eliminar carpetas obsoletas para liberar espacio en disco
    dirs_to_clean = [
        os.path.join(data_dir, 'raw'),
        os.path.join(data_dir, 'raw_backup'),
        os.path.join(data_dir, 'raw_new'),
        os.path.join(data_dir, 'processed'),
        os.path.join(data_dir, 'reduced'),
        os.path.join(data_dir, 'train'),
        os.path.join(data_dir, 'test')
    ]
    
    print("[~] Limpiando carpetas de datos anteriores para liberar espacio...")
    for d in dirs_to_clean:
        if os.path.exists(d):
            print(f"  [-] Eliminando: {os.path.relpath(d, project_root)}")
            shutil.rmtree(d, ignore_errors=True)
            
    # 2. Crear nuevos directorios de destino
    train_dest = os.path.join(data_dir, 'train')
    test_dest = os.path.join(data_dir, 'test')
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)
    
    if not os.path.exists(dataset_src):
        raise FileNotFoundError(f"No se encontró la carpeta Dataset de origen en: {dataset_src}")
        
    print(f"[~] Iniciando la reorganización desde '{os.path.relpath(dataset_src, project_root)}'...")
    
    # Obtener y ordenar las clases (carpetas A-Z)
    classes = sorted([d for d in os.listdir(dataset_src) if os.path.isdir(os.path.join(dataset_src, d))])
    
    for cls in classes:
        cls_dir = os.path.join(dataset_src, cls)
        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"  Clase '{cls}': {len(files)} imágenes encontradas.")
        
        # Crear subcarpeta para el entrenamiento
        cls_train_dest = os.path.join(train_dest, cls)
        os.makedirs(cls_train_dest, exist_ok=True)
        
        # Dividir: 180 para entrenamiento y el resto (20) para pruebas
        train_files = files[:180]
        test_files = files[180:]
        
        # Copiar archivos de entrenamiento
        for f in train_files:
            shutil.copy2(os.path.join(cls_dir, f), os.path.join(cls_train_dest, f))
            
        # Copiar y renombrar archivos de prueba de forma plana
        for idx, f in enumerate(test_files):
            ext = os.path.splitext(f)[1]
            new_name = f"{cls}_test_{idx+1}{ext}"
            shutil.copy2(os.path.join(cls_dir, f), os.path.join(test_dest, new_name))
            
    print("[+] Reorganización y limpieza completadas exitosamente!")

if __name__ == '__main__':
    reorganize()
