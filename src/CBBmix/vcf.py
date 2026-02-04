from cyvcf2 import VCF
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


def read_genotypes(genotype: list):
    """Shared helper function for genotype classification."""
    allele1, allele2, _ = genotype
    if all([k == 1 for k in (allele1, allele2)]):
        return 'homalt'
    elif any([k == 1 for k in (allele1, allele2)]):
        return 'hetalt'
    else:
        return 'skip'


class ChromosomeArmLookup:
    def __init__(self, data):
        self.centromeres = {}
        self.chr_ends = {}
        
        for row in data:
            chrom, start, end, arm = row
            if chrom not in self.centromeres:
                self.centromeres[chrom] = None
                self.chr_ends[chrom] = 0
            
            if arm == 'p':
                self.centromeres[chrom] = end
            self.chr_ends[chrom] = max(self.chr_ends[chrom], end)
    
    def query(self, chrom, pos):
        if chrom not in self.centromeres:
            return None
        
        centromere = self.centromeres[chrom]
        return 'p' if pos < centromere else 'q'
    
    def query_array(self, chroms, positions):
        chroms = np.asarray(chroms)
        positions = np.asarray(positions)
        centromere_positions = np.array([
            self.centromeres.get(c, np.nan) for c in chroms
        ])        
        result = np.where(positions < centromere_positions, 'p', 'q')
        unknown_mask = np.array([c not in self.centromeres for c in chroms])
        result[unknown_mask] = None
        
        return result

        
class SomaticVariantCollector:
    def __init__(self, vcf):
        self._chrs_arms_lookup = ChromosomeArmLookup(_CHROMOSOME_ARMS)
        self.vcf_file = VCF(vcf)
        self._som_threshold = .5
        self._csq_keys = [
            j.strip() for j in self.vcf_file.get_header_type('CSQ')['Description'].replace('"','').split('Format: ')[1].split('|')
        ] 
        self.somatic_vars = self._collect_somatic_vars() 

    def _collect_somatic_vars(self):
        som_vars = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for variant in self.vcf_file:
            som_prob = variant.INFO.get('somProb')
            if som_prob is not None and som_prob > self._som_threshold:
                arm = self._chrs_arms_lookup.query(
                            variant.CHROM, int(variant.POS))
                som_vars[variant.CHROM][arm]['DP'].append(
                    variant.gt_depths[0]
                )
                som_vars[variant.CHROM][arm]['alt_DP'].append(
                    variant.gt_alt_depths[0])
                som_vars[variant.CHROM][arm]['VAF'].append(
                    variant.gt_alt_freqs[0])
        return som_vars


class GermlineVariantCollector:
    def __init__(self, vcf, af_thresholds=[0.25, 0.75]):
        self._chrs_arms_lookup = ChromosomeArmLookup(_CHROMOSOME_ARMS)
        self.vcf_file = VCF(vcf)
        self._af_thresholds = af_thresholds
        self._csq_keys = [
            j.strip() for j in self.vcf_file.get_header_type('CSQ')['Description'].replace('"','').split('Format: ')[1].split('|')
        ] 
        self.germline_vars = self._collect_germline_vars() 

    def _collect_germline_vars(self):
        germ_vars = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        for variant in self.vcf_file:
            # let's take advantage of ENEO annotation. 
            # as we want only stuff that we could trust, we couldn't go 
            # too much away from a boundary over the 0.5 median
            hetprob = variant.INFO.get('hetProb') 
            af = variant.gt_alt_freqs[0]
            if all([hetprob > 0.5, af >= self._af_thresholds[0], af <= self._af_thresholds[1]]):
                arm = self._chrs_arms_lookup.query(
                    variant.CHROM, int(variant.POS)
                )
                # TODO: this last block should removed as we're interested just in het variants.
                genotype = read_genotypes(variant.genotypes[0])
                if genotype != 'skip':
                    germ_vars[variant.CHROM][arm][genotype]['DP'].append(
                        variant.gt_depths[0]
                    )
                    germ_vars[variant.CHROM][arm][genotype]['alt_DP'].append(
                        variant.gt_alt_depths[0])
                    germ_vars[variant.CHROM][arm][genotype]['VAF'].append(
                        variant.gt_alt_freqs[0])
        return germ_vars