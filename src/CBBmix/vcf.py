import cyvcf2.VCF as VCF
import numpy as np
from collections import defaultdict


_CHROMOSOME_ARMS = [
    ('chr1', 0, 123400000, 'p'), ('chr1', 123400000, 248956422, 'q'),
    ('chr10', 0, 39800000, 'p'), ('chr10', 39800000, 133797422, 'q'),
    ('chr11', 0, 53400000, 'p'), ('chr11', 53400000, 135086622, 'q'),
    ('chr12', 0, 35500000, 'p'), ('chr12', 35500000, 133275309, 'q'),
    ('chr13', 0, 17700000, 'p'), ('chr13', 17700000, 114364328, 'q'),
    ('chr14', 0, 17200000, 'p'), ('chr14', 17200000, 107043718, 'q'),
    ('chr15', 0, 19000000, 'p'), ('chr15', 19000000, 101991189, 'q'),
    ('chr16', 0, 36800000, 'p'), ('chr16', 36800000, 90338345, 'q'),
    ('chr17', 0, 25100000, 'p'), ('chr17', 25100000, 83257441, 'q'),
    ('chr18', 0, 18500000, 'p'), ('chr18', 18500000, 80373285, 'q'),
    ('chr19', 0, 26200000, 'p'), ('chr19', 26200000, 58617616, 'q'),
    ('chr2', 0, 93900000, 'p'), ('chr2', 93900000, 242193529, 'q'),
    ('chr20', 0, 28100000, 'p'), ('chr20', 28100000, 64444167, 'q'),
    ('chr21', 0, 12000000, 'p'), ('chr21', 12000000, 46709983, 'q'),
    ('chr22', 0, 15000000, 'p'), ('chr22', 15000000, 50818468, 'q'),
    ('chr3', 0, 90900000, 'p'), ('chr3', 90900000, 198295559, 'q'),
    ('chr4', 0, 50000000, 'p'), ('chr4', 50000000, 190214555, 'q'),
    ('chr5', 0, 48800000, 'p'), ('chr5', 48800000, 181538259, 'q'),
    ('chr6', 0, 59800000, 'p'), ('chr6', 59800000, 170805979, 'q'),
    ('chr7', 0, 60100000, 'p'), ('chr7', 60100000, 159345973, 'q'),
    ('chr8', 0, 45200000, 'p'), ('chr8', 45200000, 145138636, 'q'),
    ('chr9', 0, 43000000, 'p'), ('chr9', 43000000, 138394717, 'q'),
    ('chrX', 0, 61000000, 'p'), ('chrX', 61000000, 156040895, 'q'),
    ('chrY', 0, 10400000, 'p'), ('chrY', 10400000, 57227415, 'q'),
]


class ChromosomeArmLookup:
    def __init__(self, data):
        """
        Initialize the lookup structure from chromosome arm data.
        
        Parameters:
        -----------
        data : list of tuples or DataFrame-like
            Each entry should have (chr, start, end, arm)
        """
        # Build a dictionary mapping chromosome to (centromere_position, chr_end)
        # The centromere is where p arm ends and q arm begins
        self.centromeres = {}
        self.chr_ends = {}
        
        for row in data:
            chrom, start, end, arm = row
            if chrom not in self.centromeres:
                self.centromeres[chrom] = None
                self.chr_ends[chrom] = 0
            
            if arm == 'p':
                self.centromeres[chrom] = end  # p arm ends at centromere
            self.chr_ends[chrom] = max(self.chr_ends[chrom], end)
    
    def query(self, chrom, pos):
        """
        Query a single position.
        
        Parameters:
        -----------
        chrom : str
            Chromosome name (e.g., 'chr1')
        pos : int
            Genomic position
            
        Returns:
        --------
        str : 'p' or 'q' (or None if chromosome not found)
        """
        if chrom not in self.centromeres:
            return None
        
        centromere = self.centromeres[chrom]
        return 'p' if pos < centromere else 'q'
    
    def query_array(self, chroms, positions):
        """
        Query multiple positions efficiently using numpy.
        
        Parameters:
        -----------
        chroms : array-like
            Array of chromosome names
        positions : array-like
            Array of genomic positions
            
        Returns:
        --------
        numpy array of 'p' or 'q' values
        """
        chroms = np.asarray(chroms)
        positions = np.asarray(positions)
        centromere_positions = np.array([
            self.centromeres.get(c, np.nan) for c in chroms
        ])        
        result = np.where(positions < centromere_positions, 'p', 'q')
        unknown_mask = np.array([c not in self.centromeres for c in chroms])
        result[unknown_mask] = None
        
        return result

class GermlineVariantCollector:
    def __init__(self, vcf):
        self._chrs_arms_lookup = ChromosomeArmLookup(_CHROMOSOME_ARMS)
        self.vcf_file = VCF(vcf)
        self._common_germ_af = .05
        self.germline_vars = self._collect_germline_vars() 
        self._csq_keys = [
            j.strip() for j in self.vcf_file.get_header_type('CSQ')['Description'].replace('"','').split('Format: ')[1].split('|')
        ] 

    @staticmethod
    def read_genotypes(genotype: list):
        allele1, allele2, _ = genotype
        if all([k == 1 for k in (allele1, allele2)]):
            return 'homalt'
        elif any([k==1 for k in (allele1, allele2)]):
            return 'hetalt'
        else:
            # that's a placeholder to skip homref and strange cases
            return 'skip'

    
    def _collect_germline_vars(self):
        # collect just common vars. 
        germ_vars = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for variant in self.vcf_file:
            if variant.INFO.get('CSQ'):
                csq = dict(zip(self._csq_keys, variant.INFO.get('CSQ').split('|')))
                af = csq.get('AF')
                if af is not None and af != '':
                    if float(af) > self._common_germ_af:
                        arm = self._chrs_arms_lookup.query(
                            variant.CHROM, int(variant.POS)
                        )
                        genotype = self.read_genotypes(variant.genotypes[0])
                        if genotype != 'skip':
                            germ_vars[variant.CHROM][arm][genotype].append({
                                'DP': variant.gt_depths[0],
                                'alt_DP': variant.gt_alt_depths[0],
                                'VAF': variant.gt_alt_freqs[0]
                            })
        return germ_vars


