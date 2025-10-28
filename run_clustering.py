import os
import re
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix 
from Bio import SeqIO
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
import warnings


FASTA_FILE = "sequencias.fasta" 
LABELS_FILE = "labels.csv"      
N_COMPONENTS_PCA = 300
KMER_K1 = 2    
KMER_SKIP = 1   
KMER_K2 = 2     
MAX_CLUSTERS_TO_TEST = 15       
OUTPUT_RESULTS_FILE = "cluster_results.csv"
OUTPUT_CORR_FILE = "metrics_correlation.csv"


VALID_AA_REGEX = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")

def load_data(fasta_file, labels_file):
    """ Carrega sequências e rótulos, alinhando-os. """
    print(f"Carregando sequências de {fasta_file}...")
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        clean_seq = VALID_AA_REGEX.sub("", str(record.seq).upper())
        sequences[record.id] = clean_seq

    labels_df = pd.read_csv(labels_file)
    label_map = dict(zip(labels_df['id'], labels_df['label']))

    ordered_seqs = []
    ordered_labels = []
    
    for seq_id, seq in sequences.items():
      
        main_id = seq_id.split('_')[0] 
        if main_id in label_map:
            ordered_seqs.append(seq)
            ordered_labels.append(label_map[main_id])
        else:
            # Tenta o ID completo
             if seq_id in label_map:
                ordered_seqs.append(seq)
                ordered_labels.append(label_map[seq_id])

    if not ordered_seqs:
        print("Erro: Nenhum dado de sequência e rótulo correspondente foi encontrado.")
        return None, None

    le = LabelEncoder()
    y_true = le.fit_transform(ordered_labels)
    
    print(f"Carregamento concluído: {len(ordered_seqs)} sequências alinhadas com rótulos.")
    return ordered_seqs, y_true

def get_gapped_kmers(sequence, k1, skip, k2):
    """ Extrai k-mers gapeados 2x1x2 de uma única sequência. """
    kmers = set()
    window_size = k1 + skip + k2
    if len(sequence) < window_size:
        return kmers
        
    for i in range(len(sequence) - window_size + 1):
   
        kmer = sequence[i : i+k1] + "x" + sequence[i+k1+skip : i+window_size]
        kmers.add(kmer)
    return kmers

def create_binary_matrix(sequences, k1, skip, k2):
    """ Cria a matriz binária de presença/ausência de k-mers usando matriz esparsa. """
    print("Iniciando extração de features (k-mers gapeados 2X1X2)...")
    all_kmers_set = set()
    sequence_kmers_list = []

    for seq in sequences:
        kmers = get_gapped_kmers(seq, k1, skip, k2)
        sequence_kmers_list.append(kmers)
        all_kmers_set.update(kmers)
    
    if not all_kmers_set:
        print("Erro: Nenhum k-mer foi extraído.")
        return None, None

    feature_list = sorted(list(all_kmers_set))
    kmer_to_index = {kmer: i for i, kmer in enumerate(feature_list)}
    
    n_sequences = len(sequences)
    n_features = len(feature_list)
    print(f"Total de k-mers únicos (features): {n_features}")


    row_ind, col_ind, data = [], [], []
    
    for i, seq_kmers in enumerate(sequence_kmers_list):
        for kmer in seq_kmers:
            if kmer in kmer_to_index:
                j = kmer_to_index[kmer]
                row_ind.append(i)
                col_ind.append(j)
                data.append(1)
    
    
    binary_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_sequences, n_features))
    
    return binary_matrix, feature_list

def run_pca_projection(matrix, n_components):
    """ Projeta a matriz em uma base TruncatedSVD (similar ao PCA para sparse data). """
    
    
    n_components = min(n_components, matrix.shape[0] - 1, matrix.shape[1])
    print(f"Rodando TruncatedSVD para {n_components} componentes...")
    
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    matrix_pca = svd.fit_transform(matrix) # Mantenha o nome 'matrix_pca'
    
    explained_variance = np.sum(svd.explained_variance_ratio_)
    print(f"TruncatedSVD concluído. Variância explicada: {explained_variance:.2%}")
    
    return matrix_pca

def run_all_clustering(X_data, y_true, min_k=2, max_k=15):
    """ Roda os algoritmos de cluster, calcula métricas internas e externas. """
    print(f"Iniciando avaliação de clustering (k de {min_k} a {max_k})...")
    
    algorithms = {
        'KMeans': KMeans(n_init=10, random_state=42),
        'Agglomerative_Ward': AgglomerativeClustering(linkage='ward'),
        #'Spectral': SpectralClustering(affinity='nearest_neighbors', n_init=10, random_state=42, n_neighbors=15)#
    }
    
    results = []
    
    for k in range(min_k, max_k + 1):
        for alg_name, model_template in algorithms.items():
            model = model_template
            model.set_params(n_clusters=k)
            
            try:
              
                labels_pred = model.fit_predict(X_data)
                
             
                silhouette = silhouette_score(X_data, labels_pred)
                davies_bouldin = davies_bouldin_score(X_data, labels_pred)
                calinski_harabasz = calinski_harabasz_score(X_data, labels_pred)
                
              
                ari = adjusted_rand_score(y_true, labels_pred)
                nmi = normalized_mutual_info_score(y_true, labels_pred)
                
                results.append({
                    'algorithm': alg_name,
                    'k': k,
                    'silhouette': silhouette,
                    'davies_bouldin': davies_bouldin,
                    'calinski_harabasz': calinski_harabasz,
                    'ari': ari,
                    'nmi': nmi
                })
                
            except Exception:
                pass
                
    return pd.DataFrame(results)

def find_best_configuration(results_df):
    """ Analisa os resultados e sugere a melhor configuração. """
    if results_df.empty:
        return

    print("\n--- Análise de Correlação (Métricas Internas vs. ARI/NMI) ---")
    
    internal_metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    external_metrics = ['ari', 'nmi']
    
    correlation_matrix = results_df[internal_metrics + external_metrics].corr()
    
    print("Correlação entre Internas e Externas (ARI/NMI):")
    print(correlation_matrix.loc[internal_metrics, external_metrics].to_string())
    correlation_matrix.to_csv(OUTPUT_CORR_FILE)
    print(f"\nMatriz de correlação salva em {OUTPUT_CORR_FILE}")


    corr_with_ari = correlation_matrix['ari'][internal_metrics].abs()
    best_internal_metric = corr_with_ari.idxmax()
    
    print(f"\nMelhor Métrica Interna (mais correlacionada com ARI): {best_internal_metric} (Corr: {corr_with_ari.max():.4f})")
    
   
    if best_internal_metric == 'davies_bouldin':
        best_run = results_df.loc[results_df[best_internal_metric].idxmin()]
    else:
        best_run = results_df.loc[results_df[best_internal_metric].idxmax()]

    print("\n--- Melhor Configuração Sugerida ---")
    print(f"(Baseado na otimização da métrica interna: {best_internal_metric})")
    print(best_run.to_string())

def main():

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    
    sequences, y_true = load_data(FASTA_FILE, LABELS_FILE)
    if sequences is None: return
        
    
    binary_matrix, features = create_binary_matrix(sequences, KMER_K1, KMER_SKIP, KMER_K2)
    if binary_matrix is None: return
        
   
    matrix_pca = run_pca_projection(binary_matrix, n_components=N_COMPONENTS_PCA)
    

    results_df = run_all_clustering(matrix_pca, y_true, min_k=2, max_k=MAX_CLUSTERS_TO_TEST)
    
    if results_df.empty:
        print("Processo de clustering falhou ou não gerou resultados.")
        return
        
    
    results_df.to_csv(OUTPUT_RESULTS_FILE, index=False)
    print(f"\nResultados completos salvos em: {OUTPUT_RESULTS_FILE}")
    
   
    find_best_configuration(results_df)

if __name__ == "__main__":
    main()