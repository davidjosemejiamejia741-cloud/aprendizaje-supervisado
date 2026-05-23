import json
import os

def main():
    notebook_path = r"c:\Users\ESTUDIANTE\Downloads\aprendizaje_supervisado\proyecto_final.ipynb"
    if not os.path.exists(notebook_path):
        print(f"File not found: {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                if 'data/raw/asl_alphabet_test' in line:
                    line = line.replace('data/raw/asl_alphabet_test', 'data/test')
                    modified = True
                new_source.append(line)
            cell['source'] = new_source

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Successfully updated proyecto_final.ipynb")
    else:
        print("No changes needed or found in proyecto_final.ipynb")

if __name__ == '__main__':
    main()
