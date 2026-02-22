import numpy as np
import os
from CBBmix.baf import compute_baf
from CBBmix.utils import ChromosomeArmLookup, _CHROMOSOME_ARMS
from collections import defaultdict
import numpyro.distributions as dist
import jax.numpy as jnp
from multiprocessing import Pool

def _worker_baf(args):
    """
    Top-level helper function for multiprocessing.
    args: (gene_id, chrom, arm, bam_path, ref_path, region_string)
    """
    gid, chrom, arm, bam, ref, region = args
    # Call the static method logic
    pos, depths, alts = GeneEntry._get_BAFs(bam, ref, region)
    return gid, chrom, arm, (pos, depths, alts)


class GTF_record(object):
    """
    A GTF record is the first building block of the parser.
    The attribute field is parsed resulting in a dict.

    Attributes
    ----------
    chromosome : str
        The chromosome that the GTF record belongs to.
    source : str
        The source of the GTF record.
    feature_type : str
        The type of feature that the GTF record represents.
    start : int
        The start position of the feature in the chromosome.
    end : int
        The end position of the feature in the chromosome.
    score : str
        The score of the GTF record.
    strand : str
        The strand that the GTF record belongs to.
    phase : str
        The phase of the GTF record.
    length : int
        The length of the feature.
    attributes : dict
        A dictionary of attributes from the GTF record.

    Methods
    -------
    __init__(chromosome, source, feature_type, start, end, score, strand, phase, attributes)
        Initialize a GTF record object.
    parse_attributes(attributes)
        Parse the attributes of a GTF record.
    is_coding(feat_dict)
        Check if a GTF record is coding.
    """

    def __init__(
        self,
        chromosome,
        source,
        feature_type,
        start,
        end,
        score,
        strand,
        phase,
        attributes,
    ):
        self.chromosome = str(chromosome)
        self.source = source
        self.feature_type = feature_type
        self.start = int(start)
        self.end = int(end)
        self.score = score
        self.strand = strand
        self.phase = phase
        self.length = abs(self.end - self.start)
        self.attributes = GTF_record.parse_attributes(attributes)

    @staticmethod
    def parse_attributes(attributes):
        if isinstance(attributes, dict):
            return attributes
        else:
            feat_dict = {}
            keyVal = attributes.split(";")[:-1]
            for item in keyVal:
                replItem = item.replace(' "', '="')
                # populate the dict
                try:
                    k, v = replItem.split("=")
                    rk = k.replace(" ", "")
                    vk = v.replace('"', "")
                    feat_dict[rk] = vk
                except ValueError:
                    print(f"Unable to parse this attribute field\n{attributes}")
                    exit()
            return feat_dict

class GeneEntry:
    def __init__(self, id: str, chrom: str, start: int, end: int):
        self.id = id
        self.chrom = chrom
        self.start = start
        self.end = end
        self.region = f"{self.chrom}:{self.start}-{self.end}"

    @staticmethod
    def _get_BAFs(bam: str, ref: str, region: str, **kwargs):
        pos, depths, alts = compute_baf(bam, ref, region)
        return pos, depths, alts
    

class GeneCollector(object):
    def __init__(self, gtf: str, chromArms: ChromosomeArmLookup, bam: str, ref: str, threads: int = 4):
        self._gtf = gtf
        self._ref = ref
        self._bam = bam
        self._chrArms = ChromosomeArmLookup(_CHROMOSOME_ARMS)
        self.genes = self._collect_genes()
        self.bafs = self._get_bafs()
    
    def _collect_genes(self) -> dict:
        # genes is a nested chrom[arm][id] = GeneEntry
        genes = defaultdict(lambda: defaultdict(list))
        with open(self._gtf, 'r') as gtf:
            for line in gtf:
                if not line.startswith('#'):
                    entry = GTF_record(*line.rstrip().split('\t'))
                    if entry.feature_type == 'gene':
                        gene = GeneEntry(
                            entry.attributes.get('gene_name', None),
                            entry.chromosome, entry.start, entry.end
                        )
                        chromArm = self._chrArms.query(gene.chrom, gene.start)
                        genes[gene.chrom][chromArm].append(gene)
        return genes
    
    def _get_bafs(self) -> tuple:
        """
        Parallelizes BAF computation over all collected genes.
        Returns a dictionary structure: bafs[chrom][arm][gene_id] = (pos, depths, alts)
        """
        tasks = []
        for chrom, arms in self.genes.items():
            for arm, gene_list in arms.items():
                for gene in gene_list:
                    # Construct region string 'chrom:start-end'
                    region = f"{gene.chrom}:{gene.start}-{gene.end}"
                    # Pack all necessary data into a tuple
                    task = (gene.id, gene.chrom, arm, self._bam, self._ref, region)
                    tasks.append(task)
    
        results = []
        if self.threads > 1:
            with Pool(processes=self.threads) as pool:
                results = pool.map(_worker_baf, tasks)
        else:
            # Serial execution fallback
            results = [_worker_baf(t) for t in tasks]
        # mimick the same structure of the genecollector
        bafs = defaultdict(lambda: defaultdict(dict))
        for gid, chrom, arm, data in results:
            bafs[chrom][arm][gid] = data
        return bafs
    



    
    
    


        











