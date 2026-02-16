import pytest
import numpy as np
from unittest.mock import Mock, patch

from CBBmix.vcf import (
    ChromosomeArmLookup,
    GermlineVariantCollector,
    SomaticVariantCollector,
    _CHROMOSOME_ARMS,
    read_genotypes,
)


class TestChromosomeArmLookup:
    """Tests for ChromosomeArmLookup."""

    @pytest.fixture
    def lookup(self):
        return ChromosomeArmLookup(_CHROMOSOME_ARMS)

    def test_query_p_arm(self, lookup):
        """Test query returns 'p' for positions before centromere."""
        assert lookup.query('chr1', 100000000) == 'p'
        assert lookup.query('chr1', 0) == 'p'
        assert lookup.query('chr2', 50000000) == 'p'

    def test_query_q_arm(self, lookup):
        """Test query returns 'q' for positions at/after centromere."""
        assert lookup.query('chr1', 123400000) == 'q'  # exactly at centromere
        assert lookup.query('chr1', 200000000) == 'q'
        assert lookup.query('chr2', 100000000) == 'q'

    def test_query_unknown_chrom(self, lookup):
        """Test query returns None for unknown chromosome."""
        assert lookup.query('chrUn', 1000) is None
        assert lookup.query('chr99', 1000) is None

    def test_query_array(self, lookup):
        """Test batch query."""
        chroms = ['chr1', 'chr1', 'chr2', 'chrX']
        positions = [50000000, 150000000, 100000000, 30000000]

        result = lookup.query_array(chroms, positions)

        expected = np.array(['p', 'q', 'q', 'p'])
        np.testing.assert_array_equal(result, expected)

    def test_query_array_with_unknown(self, lookup):
        """Test batch query with unknown chromosome."""
        chroms = ['chr1', 'chrUn']
        positions = [50000000, 1000]

        result = lookup.query_array(chroms, positions)

        assert result[0] == 'p'
        assert result[1] is None

    def test_all_chromosomes_have_centromere(self, lookup):
        """Test all standard chromosomes are present."""
        expected_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
        for chrom in expected_chroms:
            assert chrom in lookup.centromeres
            assert lookup.centromeres[chrom] is not None


class TestReadGenotypes:
    """Tests for read_genotypes function."""

    def test_homalt(self):
        """Test homozygous alt detection."""
        assert read_genotypes([1, 1, False]) == 'homalt'

    def test_hetalt(self):
        """Test heterozygous detection."""
        assert read_genotypes([0, 1, False]) == 'hetalt'
        assert read_genotypes([1, 0, False]) == 'hetalt'

    def test_skip(self):
        """Test homref and other cases are skipped."""
        assert read_genotypes([0, 0, False]) == 'skip'
        assert read_genotypes([0, 2, False]) == 'skip'


class TestGermlineVariantCollector:
    """Tests for GermlineVariantCollector."""

    @pytest.fixture
    def mock_vcf_instance(self):
        """Create a mock VCF instance."""
        vcf = Mock()
        vcf.get_header_type = Mock(return_value={
            'Description': '"Format: A|B|C|AF|D"'
        })
        return vcf

    def test_collects_pos(self):
        """Test that POS field is collected."""
        vcf = Mock()
        vcf.get_header_type = Mock(return_value={
            'Description': '"Format: gene|AF"'
        })

        variant = Mock()
        variant.CHROM = 'chr1'
        variant.POS = 50000000
        variant.INFO.get = Mock(return_value=0.8)  # hetProb
        variant.genotypes = [[0, 1, False]]  # het
        variant.gt_depths = [60]
        variant.gt_alt_depths = [30]
        variant.gt_alt_freqs = [0.5]

        vcf.__iter__ = Mock(return_value=iter([variant]))

        with patch('CBBmix.vcf.VCF', return_value=vcf):
            collector = GermlineVariantCollector('dummy.vcf')

        assert 'chr1' in collector.germline_vars
        assert 'p' in collector.germline_vars['chr1']
        assert 'hetalt' in collector.germline_vars['chr1']['p']
        assert 'POS' in collector.germline_vars['chr1']['p']['hetalt']
        assert collector.germline_vars['chr1']['p']['hetalt']['POS'][0] == 50000000

    def test_get_chromosome_data(self):
        """Test get_chromosome_data method."""
        vcf = Mock()
        vcf.get_header_type = Mock(return_value={
            'Description': '"Format: gene|AF"'
        })

        # Create multiple variants
        variants = []
        for i, (pos, dp, alt_dp) in enumerate([
            (60000000, 50, 25),
            (50000000, 60, 28),  # Out of order
            (70000000, 70, 35),
        ]):
            v = Mock()
            v.CHROM = 'chr1'
            v.POS = pos
            v.INFO.get = Mock(return_value=0.9)
            v.genotypes = [[0, 1, False]]
            v.gt_depths = [dp]
            v.gt_alt_depths = [alt_dp]
            v.gt_alt_freqs = [alt_dp / dp]
            variants.append(v)

        vcf.__iter__ = Mock(return_value=iter(variants))

        with patch('CBBmix.vcf.VCF', return_value=vcf):
            collector = GermlineVariantCollector('dummy.vcf')

        positions, depths, alt_counts = collector.get_chromosome_data('chr1')

        # Should be sorted by position
        assert len(positions) == 3
        assert positions[0] < positions[1] < positions[2]
        np.testing.assert_array_equal(positions, [50000000, 60000000, 70000000])

    def test_get_chromosome_data_empty(self):
        """Test get_chromosome_data for missing chromosome."""
        vcf = Mock()
        vcf.get_header_type = Mock(return_value={
            'Description': '"Format: gene|AF"'
        })
        vcf.__iter__ = Mock(return_value=iter([]))

        with patch('CBBmix.vcf.VCF', return_value=vcf):
            collector = GermlineVariantCollector('dummy.vcf')

        positions, depths, alt_counts = collector.get_chromosome_data('chr99')

        assert len(positions) == 0
        assert len(depths) == 0
        assert len(alt_counts) == 0

    def test_get_available_chromosomes(self):
        """Test get_available_chromosomes method."""
        vcf = Mock()
        vcf.get_header_type = Mock(return_value={
            'Description': '"Format: gene|AF"'
        })

        variants = []
        for chrom in ['chr1', 'chr2', 'chr1']:
            v = Mock()
            v.CHROM = chrom
            v.POS = 50000000
            v.INFO.get = Mock(return_value=0.9)
            v.genotypes = [[0, 1, False]]
            v.gt_depths = [50]
            v.gt_alt_depths = [25]
            v.gt_alt_freqs = [0.5]
            variants.append(v)

        vcf.__iter__ = Mock(return_value=iter(variants))

        with patch('CBBmix.vcf.VCF', return_value=vcf):
            collector = GermlineVariantCollector('dummy.vcf')

        chroms = collector.get_available_chromosomes()

        assert 'chr1' in chroms
        assert 'chr2' in chroms
        assert len(chroms) == 2

    def test_af_threshold_filtering(self):
        """Test that AF thresholds are applied."""
        vcf = Mock()
        vcf.get_header_type = Mock(return_value={
            'Description': '"Format: gene|AF"'
        })

        variants = []
        for af in [0.1, 0.5, 0.9]:  # Only 0.5 should pass default thresholds
            v = Mock()
            v.CHROM = 'chr1'
            v.POS = 50000000
            v.INFO.get = Mock(return_value=0.9)  # hetProb
            v.genotypes = [[0, 1, False]]
            v.gt_depths = [50]
            v.gt_alt_depths = [int(50 * af)]
            v.gt_alt_freqs = [af]
            variants.append(v)

        vcf.__iter__ = Mock(return_value=iter(variants))

        with patch('CBBmix.vcf.VCF', return_value=vcf):
            collector = GermlineVariantCollector('dummy.vcf', af_thresholds=[0.25, 0.75])

        # Only the variant with AF=0.5 should be collected
        if 'chr1' in collector.germline_vars and 'p' in collector.germline_vars['chr1']:
            n_vars = len(collector.germline_vars['chr1']['p'].get('hetalt', {}).get('DP', []))
            assert n_vars == 1


class TestSomaticVariantCollector:
    """Tests for SomaticVariantCollector."""

    def test_collects_pos(self):
        """Test that POS field is collected for somatic variants."""
        vcf = Mock()
        vcf.get_header_type = Mock(return_value={
            'Description': '"Format: gene|AF"'
        })

        variant = Mock()
        variant.CHROM = 'chr1'
        variant.POS = 50000000
        variant.INFO.get = Mock(return_value=0.8)  # somProb
        variant.gt_depths = [60]
        variant.gt_alt_depths = [15]
        variant.gt_alt_freqs = [0.25]

        vcf.__iter__ = Mock(return_value=iter([variant]))

        with patch('CBBmix.vcf.VCF', return_value=vcf):
            collector = SomaticVariantCollector('dummy.vcf')

        assert 'chr1' in collector.somatic_vars
        assert 'p' in collector.somatic_vars['chr1']
        assert 'POS' in collector.somatic_vars['chr1']['p']
        assert collector.somatic_vars['chr1']['p']['POS'][0] == 50000000

    def test_somatic_prob_threshold(self):
        """Test that somProb threshold is applied."""
        vcf = Mock()
        vcf.get_header_type = Mock(return_value={
            'Description': '"Format: gene|AF"'
        })

        variants = []
        for prob in [0.3, 0.7]:  # Only 0.7 should pass
            v = Mock()
            v.CHROM = 'chr1'
            v.POS = 50000000
            v.INFO.get = Mock(return_value=prob)
            v.gt_depths = [60]
            v.gt_alt_depths = [15]
            v.gt_alt_freqs = [0.25]
            variants.append(v)

        vcf.__iter__ = Mock(return_value=iter(variants))

        with patch('CBBmix.vcf.VCF', return_value=vcf):
            collector = SomaticVariantCollector('dummy.vcf')

        # Only variant with somProb=0.7 should be collected
        n_vars = len(collector.somatic_vars.get('chr1', {}).get('p', {}).get('DP', []))
        assert n_vars == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
