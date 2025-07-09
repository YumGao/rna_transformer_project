import itertools

def generate_all_kmers(k, alphabet=['A', 'C', 'G', 'U']):

    kmers = []
    for combo in itertools.product(alphabet, repeat=k):
        kmers.append(''.join(combo))
        
    return kmers

def kmer_frequency(sequence, k):

    all_kmers = generate_all_kmers(k)
    kmer_count = {kmer: 0 for kmer in all_kmers}
    n = len(sequence)
    for i in range(n - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_count:
            kmer_count[kmer] += 1
    total_kmers = sum(kmer_count.values())
    if total_kmers > 0:
        for kmer in kmer_count:
            kmer_count[kmer] /= total_kmers
            
    return kmer_count # return dict

def sequence_to_kmer_features(sequence, k):

    kmer_freq = kmer_frequency(sequence, k)
    all_kmers = sorted(kmer_freq.keys())
    feature_vector = [kmer_freq[kmer] for kmer in all_kmers]
    
    return feature_vector # return list


# +
# demo
sequence = "AUGCAGUACGAUGCCUAUACCGAUCGAUAGC"
k = 2

feature_dict = kmer_frequency(sequence, k)
feature_vector = sequence_to_kmer_features(sequence, k)
print(feature_dict)
print(feature_vector)
# -


