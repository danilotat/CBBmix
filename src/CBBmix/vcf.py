from cyvcf2 import VCF
import numpy as np
from collections import defaultdict
from .utils import _CHROMOSOME_ARMS, ChromosomeArmLookup


_FILTERS_TO_EXCLUDE = ['PoN', 'GIAB']

def read_genotypes(genotype: list):
    """Shared helper function for genotype classification."""
    allele1, allele2, _ = genotype
    if all([k == 1 for k in (allele1, allele2)]):
        return 'homalt'
    elif any([k == 1 for k in (allele1, allele2)]):
        return 'hetalt'
    else:
        return 'skip'
        
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
                som_vars[variant.CHROM][arm]['POS'].append(
                    int(variant.POS))
        return som_vars


class GermlineVariantCollector:
    def __init__(self, vcf, filters_to_exclude=_FILTERS_TO_EXCLUDE, af_thresholds=[0.35, 0.65], min_dp=10):
        self._chrs_arms_lookup = ChromosomeArmLookup(_CHROMOSOME_ARMS)
        self.vcf_file = VCF(vcf)
        self._af_thresholds = af_thresholds
        self._filters_to_exclude = filters_to_exclude
        self._csq_keys = [
            j.strip() for j in self.vcf_file.get_header_type('CSQ')['Description'].replace('"','').split('Format: ')[1].split('|')
        ]
        # Locate the SYMBOL field in CSQ for gene-level aggregation
        try:
            self._gene_csq_idx = self._csq_keys.index('SYMBOL')
        except ValueError:
            self._gene_csq_idx = None
        self.germline_vars = self._collect_germline_vars()

    def _collect_germline_vars(self):
        germ_vars = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for variant in self.vcf_file:
            # let's take advantage of ENEO annotation. 
            # as we want only stuff that we could trust, we couldn't go 
            # too much away from a boundary over the 0.5 median
            hetprob = variant.INFO.get('hetProb') 
            af = variant.gt_alt_freqs[0]
            if all(
                [
                    hetprob > 0.5,
                    af >= self._af_thresholds[0],
                    af <= self._af_thresholds[1],
                    variant.FILTER not in self._filters_to_exclude,
                    variant.gt_depths[0] > 10
                ]):
                arm = self._chrs_arms_lookup.query(
                    variant.CHROM, int(variant.POS)
                )
                genotype = read_genotypes(variant.genotypes[0])
                if genotype == 'hetalt':
                    germ_vars[variant.CHROM][arm]['DP'].append(
                        variant.gt_depths[0]
                    )
                    germ_vars[variant.CHROM][arm]['alt_DP'].append(
                        variant.gt_alt_depths[0])
                    germ_vars[variant.CHROM][arm]['VAF'].append(
                        variant.gt_alt_freqs[0])
                    germ_vars[variant.CHROM][arm]['POS'].append(
                        int(variant.POS))
        return germ_vars

    def get_chromosome_data(self, chrom):
        """
        Get all heterozygous variants for a chromosome, sorted by position.

        Returns
        -------
        positions : np.ndarray
            Sorted genomic positions
        depths : np.ndarray
            Total read depths
        alt_counts : np.ndarray
            Alternate allele counts
        """
        positions = []
        depths = []
        alt_counts = []
        if chrom not in self.germline_vars:
            return np.array([]), np.array([]), np.array([])
        for arm in self.germline_vars[chrom]:
            arm_data = self.germline_vars[chrom][arm]
            positions.extend(arm_data.get('POS', []))
            depths.extend(arm_data.get('DP', []))
            alt_counts.extend(arm_data.get('alt_DP', []))
        if not positions:
            return np.array([]), np.array([]), np.array([])
        # Sort by position
        positions = np.array(positions)
        depths = np.array(depths)
        alt_counts = np.array(alt_counts)
        sort_idx = np.argsort(positions)
        return positions[sort_idx], depths[sort_idx], alt_counts[sort_idx]

    def get_available_chromosomes(self):
        """Return list of chromosomes with germline variants."""
        return list(self.germline_vars.keys())