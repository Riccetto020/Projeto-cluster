import pandas as pd
from Bio import SeqIO
import re
import os

FASTA_FILE_INPUT = "sequencias.fasta"
LABELS_FILE_OUTPUT = "labels.csv"

def extract_labels_from_fasta(fasta_file, output_csv):
    """
    Extrai o ID da sequência e o rótulo SCOP (ex: a.1.1.1) do cabeçalho FASTA.
    """
    if not os.path.exists(fasta_file):
        print(f"Erro: Arquivo FASTA '{fasta_file}' não encontrado. Por favor, garanta que o nome esteja correto.")
        return

    data = []
    # Regex para capturar o ID (tudo antes do primeiro espaço) e o rótulo SCOP (o padrão x.x.x.x)
    # Ex: >d1dlwa_ a.1.1.1 (A:) ...
    id_regex = re.compile(r"^>(\S+)")
    label_regex = re.compile(r"([a-z]\.\d+\.\d+\.\d+)")

    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_id_match = id_regex.match(record.description)
        label_match = label_regex.search(record.description)

        seq_id = seq_id_match.group(1) if seq_id_match else record.id
        label = label_match.group(1) if label_match else "UNKNOWN"

        data.append({'id': seq_id, 'label': label})
    
    if not data:
        print("Erro: Nenhuma sequência ou rótulo válido foi extraído.")
        return

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Sucesso! {len(df)} rótulos extraídos e salvos em: {output_csv}")

if __name__ == "__main__":
    # Certifique-se de que a variável FASTA_FILE_INPUT corresponde ao nome do arquivo que você enviou
    extract_labels_from_fasta(FASTA_FILE_INPUT, LABELS_FILE_OUTPUT)