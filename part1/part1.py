import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def extract_treebank_data(filepath, language):
    parsed_data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            columns = line.split('\t')
            if len(columns) >= 8:
                try:
                    token_id = int(columns[0])
                    head_id = int(columns[6])
                    relation = columns[7]
                    morph_feats = columns[5]
                    
                    dep_dist = abs(token_id - head_id) if head_id > 0 else np.nan
                    
                    parsed_data.append({
                        'Language': language,
                        'Relation': relation,
                        'Distance': dep_dist,
                        'Morphology': morph_feats
                    })
                except ValueError:
                    continue
                    
    return parsed_data

def parse_morphology(feat_string, target_feature):
    if pd.isna(feat_string) or feat_string == '_':
        return None
        
    for item in feat_string.split('|'):
        if '-' in item:
            key, val = item.split('-', 1)
            if key == target_feature and val != '_':
                return val
    return None

def main():
    print("Loading datasets")
    
    telugu_file = Path('telugu_treebank-master/iiit_hcu_intra_chunk_v1.conll')
    hindi_dir = Path('HDTB_pre_release_version-0.05/IntraChunk/CoNLL/utf')
    
    all_records = []
    
    if telugu_file.exists():
        all_records.extend(extract_treebank_data(telugu_file, 'Telugu'))
    else:
        print(f"Error: Could not find {telugu_file}")
        
    for hindi_file in hindi_dir.rglob('*.dat'):
        all_records.extend(extract_treebank_data(hindi_file, 'Hindi'))
        
    df = pd.DataFrame(all_records)
    
    for feat in ['gen', 'num', 'case']:
        df[feat] = df['Morphology'].apply(lambda x: parse_morphology(x, feat))

    print("\n" + "="*40)
    print("DATA ANALYSIS RESULTS")
    print("="*40 + "\n")

    print("1. DEPENDENCY DISTANCES")
    stats_df = df.groupby('Language')['Distance'].agg(['mean', 'median']).round(4)
    print(stats_df)
    
    plt.figure(figsize=(10, 6))
    hindi_dists = df[df['Language'] == 'Hindi']['Distance'].dropna()
    telugu_dists = df[df['Language'] == 'Telugu']['Distance'].dropna()
    
    plt.hist(telugu_dists, bins=range(0, 25), alpha=0.6, label='Telugu', color='#1f77b4', edgecolor='black')
    plt.hist(hindi_dists, bins=range(0, 25), alpha=0.6, label='Hindi', color='#ff7f0e', edgecolor='black')
    plt.title('Distribution of Dependency Distances: Hindi vs Telugu')
    plt.xlabel('Absolute Dependency Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('dependency_distances_plot.png')
    print("-> Histogram saved as 'dependency_distances_plot.png'\n")

    print("2. TOP 10 DEPENDENCY RELATIONS")
    for lang in ['Hindi', 'Telugu']:
        print(f"\n--- {lang} ---")
        top_10 = df[df['Language'] == lang]['Relation'].value_counts().head(10)
        print(top_10)


    print("\n3. RELATIVE FREQUENCIES OF SUBJECTS (k1) vs OBJECTS (k2)")
    subj_obj_df = df[df['Relation'].isin(['k1', 'k2', 'nsubj', 'obj'])]
    if not subj_obj_df.empty:
        freq_table = pd.crosstab(subj_obj_df['Language'], subj_obj_df['Relation'], normalize='index').round(4) * 100
        print(freq_table.applymap(lambda x: f"{x}%"))
    else:
        print("Note: Paninian tags (k1, k2) not found in expected format.")

    print("\n4. SIGNIFICANCE TESTING (Welch's T-Test)")
    t_stat, p_value = stats.ttest_ind(hindi_dists, telugu_dists, equal_var=False, nan_policy='omit')
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value:     {p_value:.4e}")
    if p_value < 0.05:
        print("Conclusion: The difference in dependency distances between the two languages is statistically significant.")

    print("\n5. MORPHOLOGICAL FEATURES SUMMARY")
    for feat in ['gen', 'num', 'case']:
        print(f"\nTop '{feat.upper()}' distributions:")
        feat_crosstab = pd.crosstab(df['Language'], df[feat])
        print(feat_crosstab)

if __name__ == "__main__":
    main()