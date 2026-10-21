import json
import sys
from pathlib import Path

def parse_notebook(filepath):
    out_path = filepath.replace('.ipynb', '_dump.txt')
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write(f"========== PARSING {filepath} ==========\n")
        
        for i, cell in enumerate(nb.get('cells', [])):
            cell_type = cell.get('cell_type', 'unknown')
            source = "".join(cell.get('source', []))
            out.write(f"\n--- CELL {i} ({cell_type}) ---\n")
            if cell_type == 'markdown':
                out.write(source + "\n")
            elif cell_type == 'code':
                out.write("CODE:\n")
                out.write(source + "\n")
                out.write("OUTPUTS:\n")
                for out_node in cell.get('outputs', []):
                    if out_node.get('output_type') == 'stream':
                        out.write("".join(out_node.get('text', [])) + "\n")
                    elif out_node.get('output_type') in ['execute_result', 'display_data']:
                        data = out_node.get('data', {})
                        if 'text/plain' in data:
                            out.write("".join(data['text/plain']) + "\n")
                        if 'image/png' in data:
                            out.write("[IMAGE OUTPUT ELIDED]\n")
                    elif out_node.get('output_type') == 'error':
                        out.write("ERROR: " + out_node.get('ename', '') + " " + out_node.get('evalue', '') + "\n")
                        
if __name__ == '__main__':
    for arg in sys.argv[1:]:
        parse_notebook(arg)
