
import json

def extract_notebook_content(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            print("MARKDOWN:")
            print("".join(cell['source']))
            print("-" * 20)
        elif cell['cell_type'] == 'code':
             source = "".join(cell['source'])
             # print just the first line of code cells to get an idea
             if source.strip():
                 print(f"CODE: {source.splitlines()[0]} ...")

if __name__ == "__main__":
    extract_notebook_content(r"c:\Users\wwefi\OneDrive\Υπολογιστής\synthetic-data-generation-with-gans\wgan_gp.ipynb")
