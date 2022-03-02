#!/usr/bin/env python3
import sys, os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation


# General info about the genome references and the location of the cassettes in the genomes
seq_record5 = SeqIO.read('./data/C5_refSeq.gb', "genbank")
seq_record20 = SeqIO.read('./data/C20_refSeq.gb', "genbank")
seq_record26 = SeqIO.read('./data/C26_refSeq.gb', "genbank")

genome_C5 = str(seq_record5.seq)
genome_C20 = str(seq_record20.seq)
genome_C26 = str(seq_record26.seq)

cass_par = [565510, 572124] # Present in C5 and C20
cass_3b = [780850, 787445]  # Only in C20
cass_26 = [574473, 581077]
cass_262 = [79542, 88496]

# Annotation information to work faster with dataframes
ncbi = pd.read_csv('./data/mpn_annotation.csv', sep='\t', header=None) # Genome annotation
ncbi.columns = ['gene', 'start', 'end', 'strand']
ncbi.set_index('gene', inplace=True)
gold = pd.read_csv('./data/goldsets.csv', sep='\t')  # Gold set 

posN = []
posE = []
for gene, cat in zip(gold['gene'], gold['class']):
    if cat=='E':
        posE += range(ncbi.loc[gene][0], ncbi.loc[gene][1]+1)
    else:
        posN += range(ncbi.loc[gene][0], ncbi.loc[gene][1]+1)
        
# Define the different set of positions in the genome were we will perform the analysis
posN = set(posN)
posE = set(posE)
cas1 = set(range(cass_par[0], cass_par[1]+1))
cas2 = set(range(cass_3b[0], cass_3b[1]+1))
cas3 = set(range(cass_26[0], cass_26[1]+1))
cas12 = cas1.union(cas2)
geno = set(range(1, 816395)).difference(cas1.union(cas2))

cas31 = set(range(cass_26[0], cass_26[1]+1))
cas32 = set(range(cass_262[0], cass_262[1]+1))
cas33 = cas31.union(cas32)

### SEVERAL FUNCTIONS TO PARSE THE DATAFRAMES 
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

def counthash(fil):
    c=0
    with open(fil, 'r') as fi:
        for line in fi:
            if line.startswith('#'):
                c+=1
            else:
                return c-1 


def correspondance(genome, positions):
    """
    Return variants mapped within <sten> which is a list in format [start, end]
    <genome> defines the record to use as reference
    """
    d = {}
    c = 0
    if genome==5:
        seq_record = seq_record5
    elif genome==20:
        seq_record = seq_record20
    else:
        seq_record = seq_record26
    for i in seq_record.features:
        if i.location.start in positions and i.type not in ['Polymorphism', 'misc_feature', 'Site', 'repeat_region', 'gene']:
            d[c] = i
            c+=1
    return d

def map_cassettes():  
    pos_cas1 = {i:'intergenic_cassette' for i in cas1}
    for v in correspondance(5, positions=cas1).values():
        if v.type!='CDS':
            name = v.qualifiers['label'][0]
        else:
            name = v.qualifiers['label'][0]
        for i in range(int(v.location.start), int(v.location.end)+1):
            if pos_cas1[i][0]!='i':
                pos_cas1[i]+= ' | '+name
            else:
                pos_cas1[i] = name
    pos_cas2 = {i:'intergenic_cas1' if i<700000 else 'intergenic_cas2' for i in cas12}
    for v in correspondance(20, positions=cas12).values():
        if v.type!='CDS':
            name = v.qualifiers['label'][0]
        else:
            name = v.qualifiers['label'][0]
        assigned = False
        for i in range(int(v.location.start), int(v.location.end)+1):
            if not assigned:
                if i<700000:
                    name = 'KS1-1 '+name
                else:
                    name = 'KS1-2 '+name
                assigned = True
            if pos_cas2[i][0]!='i':
                pos_cas2[i]+= ' | '+name
            else:
                pos_cas2[i] = name
    pos_cas3 = {i:'intergenic_cas1' if i<90000 else 'intergenic_cas2' for i in cas33}
    for v in correspondance(26, positions=cas33).values():
        if v.type!='CDS':
            if v.type!='RBS':
                name = v.qualifiers['label'][0]
            else:
                continue
        else:
            name = v.qualifiers['label'][0]
        assigned = False
        for i in range(int(v.location.start), int(v.location.end)+1):
            if not assigned:
                if i<90000:
                    name = 'TKS2 '+name
                else:
                    name = 'KS1 '+name
                assigned = True
            if pos_cas3[i][0]!='i':
                pos_cas3[i]+= ' | '+name
            else:
                pos_cas3[i] = name
    return {5:pos_cas1, 20:pos_cas2, 26:pos_cas3}

maps = map_cassettes()
gold_hash = {k:v for k, v in zip(list(gold['gene']), list(gold['class']))}

def process_effect(effect, position, genome=5):
    eff = str(effect).split(',')
    rs = {}
    for e in eff:
        string = e.split('|')
        if len(string)>5:
            typ, impact, gene = string[1], string[2], string[4]
            # Only keep gene variant
            if 'up' in string[1] or 'down' in string[1]:
                ann = 'intergenic'
                mut = string[-7]
            else:
                if gene in gold_hash:
                    if gold_hash[gene]=='E':
                        ann = 'essential'
                    else:
                        ann = 'non-essential'
                else:
                    ann = 'gene'
                mut = string[-6]
            if position in maps[genome]:
                ann = 'cassette'
                gene = maps[genome][position]
            if string[0] not in rs:
                rs[string[0]] = [typ, impact, gene, mut, ann]
    return rs

def simplify_effect(df, genome=5):
    """ Simplifies the field effect to be human readable """
    c = 1
    rs = {}

    for i, j in df.iterrows():
        try:
            effd = process_effect(str(j.EFF), j.POS, genome)
        except:
            print(i, j)
        for alt, altn in zip(str(j.ALT).split(','), str(j.ALTN).split(',')):
            try:
                rs[c] = [j.SAMPLE, j.P, int(j.POS), j.QUAL, j.TOT, j.REFN, int(altn), int(altn)/j.REFN*100, str(j.REF), alt]+effd[alt]
                c+=1
            except:
                pass
    new_df = pd.DataFrame.from_dict(rs, orient='index')
    new_df.columns = ['SAMPLE', 'PASS', 'POS', 'QUAL', 'TOT', 'REFN', 'ALTN', 'FRAC', 'REF', 'ALT', 'EFF', 'IMPACT', 'AFF', 'MUT', 'ANN_TYPE']
    return new_df


def prok(effect_col):
    """ Remove splice and intron signal events, not meaningful for prokaryotes"""
    return ','.join([ii for ii in effect_col.split(',') if 'splice' not in ii and 'intron' not in ii])
def orderdf(df, ordered_classes, col='SAMPLE'):
    """ To order by sample"""
    df_list = []
    for i in ordered_classes:
        df_list.append(df[df[col]==i])
    ordered_df = pd.concat(df_list)
    return ordered_df.reset_index(drop=True)


def parse_variations(gre, order, genome=5, fname='./results/data1.pickle'):
    """ Function to parse the variations removing the header information """
    if os.path.isfile(fname):
        print('Loading data for genome {}'.format(genome))
        return pd.read_pickle(fname)
    else:
        print('Processing data for genome {}'.format(genome))
        snpcalls = []
        for fil in glob.glob(gre):
            if 'filter' in gre:
                ide = fil.split('/')[2].split('.')[0]
            else:
                ide = fil.split('/')[1]
            
            df = pd.read_csv(fil, header=counthash(fil), sep='\t')    
            df['TOT'] = [int(i.split(':')[1]) for i in df[ide]]
            df['REFN'] = [int(i.split(':')[3]) for i in df[ide]]
        
            df['ALTN'] = (df['TOT']-df['REFN'])
            df['ALTS'] = [i.split(':')[5] for i in df[ide]]
            df['FRAC'] = (df['ALTN'])/df['TOT']*100
        
            df['SAMPLE'] = [ide for i in df[ide]]
            df['P'] = [int(i.split('_')[0].replace('p', '').replace('3IPTG', '18')) for i in df['SAMPLE']]
            df['EFF'] = [prok(info.split('ANN=')[1]) for info in df.INFO]
            df = df[['POS', 'REF', 'ALT', 'QUAL', 'TOT', 'REFN', 'ALTN', 'ALTS', 'FRAC', 'SAMPLE', 'P', 'EFF']]
            snpcalls.append(df)
        snpcalls = pd.concat(snpcalls).sort_values(['P', 'SAMPLE'])
        orderedf = orderdf(snpcalls, order)
        print('Cleaning data for genome {}'.format(genome))
        cleanedf, nalist = reduce_mem_usage(simplify_effect(orderedf, genome=genome))
        print('Saving data for genome {}'.format(genome))
        cleanedf.to_pickle(fname)
        return cleanedf




#### FOR RUNNING DN/DS ANALYSES
# Adaptation of https://github.com/adelq/dnds/blob/master/dnds.py to mycoplasma pneumoniae and vcf files

### This program is written for dN/dS ratio calculation based on VCF and GenBank files.
from Bio import SeqIO
from Bio.Seq import Seq
from math import log

codons = {"TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
          "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "W", "TGG": "W",
          "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
          "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
          "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
          "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
          "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
          "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"}
BASES = {'A', 'G', 'T', 'C'}
seq_record5 = SeqIO.read('./data/C5_refSeq.gb', "genbank")
seq_record20 = SeqIO.read('./data/C20_refSeq.gb', "genbank")

# copy from https://github.com/adelq/dnds/commit/783e8197541aeb456e19c8ae9effa54cc16b02c0
# copy from https://github.com/adelq/dnds/blob/master/dnds.py
def split_seq(seq, n=3):
    # Returns sequence split into chunks of n characters, default is codons
    return [seq[i:i+n] for i in range(0, len(seq), n)]

def average_list(l1, l2):
    return [float(i1 + i2)/2 for i1, i2 in zip(l1, l2)]

def dna_to_protein(codon):
    # Returns single letter amino acid code for given codon
    return codons[codon]

def translate(seq):
    # Translate a DNA sequence into the 1-letter amino acid sequence
    return "".join([dna_to_protein(codon) for codon in split_seq(seq)])

def is_synonymous(codon1, codon2):
    # Returns boolean whether given codons are synonymous
    return dna_to_protein(codon1) == dna_to_protein(codon2)

def dnds_codon(codon):
    # Returns list of synonymous counts
    syn_list = []
    for i in range(len(codon)):
        base = codon[i]
        other_bases = BASES - {base}
        syn = 0
        for new_base in other_bases:
            new_codon = codon[:i] + new_base + codon[i + 1:]
            syn += int(is_synonymous(codon, new_codon))
        syn_list.append(float(syn)/3)
    return syn_list

def dnds_codon_pair(codon1, codon2):
    # Get the dN/dS for the given codon pair
    return average_list(dnds_codon(codon1), dnds_codon(codon2))

def syn_sum(seq1, seq2):
    # Get the sum of synonymous sites from two DNA sequences
    syn = 0
    codon_list1 = split_seq(seq1)
    codon_list2 = split_seq(seq2)
    for i in range(len(codon_list1)):
        codon1 = codon_list1[i]
        codon2 = codon_list2[i]
        if len(codon2)==3:
            dnds_list = dnds_codon_pair(codon1, codon2)
            syn += sum(dnds_list)
        else:
            return float(syn)
    return float(syn)

def hamming(s1, s2):
    # Return the hamming distance between 2 DNA sequences
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2)) + abs(len(s1) - len(s2))

def codon_subs(codon1, codon2):
    # Returns number of synonymous substitutions in provided codon pair Methodology for multiple substitutions from Dr. Swanson, UWashington https://faculty.washington.edu/wjs18/dnds.ppt
    diff = hamming(codon1, codon2)
    if diff < 1:
        return 0
    elif diff == 1:
        return int(translate(codon1) == translate(codon2))

    syn = 0
    for i in range(len(codon1)):
        base1 = codon1[i]
        base2 = codon2[i]
        if base1 != base2:
            new_codon = codon1[:i] + base2 + codon1[i + 1:]
            syn += int(is_synonymous(codon1, new_codon))
            syn += int(is_synonymous(codon2, new_codon))
    return float(syn) / diff

def substitutions(seq1, seq2):
    # Returns number of synonymous and nonsynonymous substitutions
    dna_changes = hamming(seq1, seq2)
    codon_list1 = split_seq(seq1)
    codon_list2 = split_seq(seq2)
    syn = 0
    for i in range(len(codon_list1)):
        codon1 = codon_list1[i]
        codon2 = codon_list2[i]
        if len(codon2)==3:
            syn += codon_subs(codon1, codon2)
        else:
            return [syn, dna_changes-syn]
    return [syn, dna_changes-syn]

def clean_sequence(seq):
    # Clean up provided sequence by removing whitespace.
    return seq.replace(' ', '')

def dnds(seq1, seq2):
    """Main function to calculate dN/dS between two DNA sequences per Nei &
    Gojobori 1986. This includes the per site conversion adapted from Jukes &
    Cantor 1967.
    """
    # Strip any whitespace from both strings
    seq1 = clean_sequence(seq1)
    seq2 = clean_sequence(seq2)
    # Check that both sequences have the same length
    # assert len(seq1) == len(seq2)
    # Check that sequences are codons
    # assert len(seq1) % 3 == 0
    syn_sites = syn_sum(seq1, seq2)
    non_sites = len(seq1)-syn_sites
    syn_subs, non_subs = substitutions(seq1, seq2)
    pn = float(non_subs)/non_sites
    ps = float(syn_subs)/syn_sites
    dn = -0.75 * log(1 - (4 * pn / 3))
    ds = -0.75 * log(1 - (4 * ps / 3))
    return round(float(pn), 3), round(float(ps), 3), round(float(dn), 3), round(float(ds), 3)

def snp_in_gene(file, record, pos):
    # Confirm whether an SNP in a gene
    output = None
    for records in SeqIO.parse(file, "genbank"):
        if record == records.id:
            for feature in records.features:
                if feature.type == 'CDS' and (feature.location.start <= pos and feature.location.end >= pos):
                    output = feature
                    break
    return output

def get_new_sequence(record, gene, pos):
    for feature in record.features:
        if ('locus_tag' in feature.qualifiers and gene== feature.qualifiers['locus_tag'][0]) or ('label' in feature.qualifiers and gene==feature.qualifiers['label'][0]):
            st = feature.location.start
            en = feature.location.end
            strand = feature.location.strand
            raw_seq = str(record.seq[st:en])
            new_seq = str(record.seq[st:pos[0]-1]) + pos[2] + str(record.seq[pos[0]+len(pos[1])-1:en])
            if feature.location.strand < 0:
                raw_seq = str(Seq(raw_seq).reverse_complement())
                new_seq = str(Seq(new_seq).reverse_complement())
                if gene=='LacI4':
                    raw_seq = raw_seq[:-4]
                    new_seq = new_seq[:-4]
            return raw_seq, new_seq
        
def evolution_of_sample(df, genome=5):
    if genome==5:
        record_seq = seq_record5
    else:
        record_seq = seq_record20    
        
    rs = {}
    for sample in set(df['SAMPLE']):
        pn, ps, pns, dn, ds, dns = 0, 0, 0, 0, 0, 0
        subdf = df[df['SAMPLE']==sample]
        for pos, ref, alt, gene in zip(list(subdf['POS']),list(subdf['REF']), list(subdf['ALT']), list(subdf['AFF'])):
            try:
                raw_seq, new_seq = get_new_sequence(record_seq, gene, pos=[pos, ref, alt])
                newpn, newps, newdn, newds = dnds(raw_seq, new_seq)
                pn += newpn
                ps += newps
                dn += newdn
                ds += newds
            except:
                pass
        if pn == 0 and ps == 0:
            pns, dns = '/', '/'
        elif ps == 0:
            pns, dns = '+', '+'
        elif pn == 0:
            pns, dns = '-', '-'
        else:
            pns, dns = pn/ps, dn/ds
        rs[sample] = [pn, ps, pns, dn, ds, dns]
    return rs