import pytest
import numpy as np
from unittest.mock import Mock, patch, PropertyMock

from CBBmix.vcf import (
    ChromosomeArmLookup,
    GermlineVariantCollector,
    _CHROMOSOME_ARMS,
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


class TestGermlineVariantCollector:
    """Tests for GermlineVariantCollector."""

    @pytest.fixture
    def mock_variant(self):
        """Create a mock variant."""
        variant = Mock()
        variant.CHROM = 'chr1'
        variant.POS = 100000000
        variant.INFO.get = Mock(return_value='A|B|C|0.1|D')
        variant.genotypes = [[1, 0, False]]  # het
        variant.gt_depths = [50]
        variant.gt_alt_depths = [25]
        variant.gt_alt_freqs = [0.5]
        return variant

    @pytest.fixture
    def mock_vcf(self, mock_variant):
        """Create a mock VCF."""
        vcf = Mock()
        vcf.__iter__ = Mock(return_value=iter([mock_variant]))
        vcf.get_header_type = Mock(return_value={
            'Description': '"Format: A|B|C|AF|D"'
        })
        return vcf

    def test_read_genotypes_homalt(self):
        """Test homozygous alt detection."""
        assert GermlineVariantCollector.read_genotypes([1, 1, False]) == 'homalt'

    def test_read_genotypes_hetalt(self):
        """Test heterozygous detection."""
        assert GermlineVariantCollector.read_genotypes([0, 1, False]) == 'hetalt'
        assert GermlineVariantCollector.read_genotypes([1, 0, False]) == 'hetalt'

    def test_read_genotypes_skip(self):
        """Test homref and other cases are skipped."""
        assert GermlineVariantCollector.read_genotypes([0, 0, False]) == 'skip'
        assert GermlineVariantCollector.read_genotypes([0, 2, False]) == 'skip'

    @patch('germline_variant_collector.VCF')
    def test_collect_germline_vars_structure(self, mock_vcf_class):
        """Test collected variants have correct nested structure."""
        # Setup mock
        mock_vcf_instance = Mock()
        mock_vcf_class.return_value = mock_vcf_instance
        mock_vcf_instance.get_header_type.return_value = {
            'Description': '"Format: gene|impact|AF"'
        }
        
        # Create mock variant with AF > 0.05
        variant = Mock()
        variant.CHROM = 'chr1'
        variant.POS = 100000000
        variant.INFO.get.return_value = 'BRCA1|HIGH|0.1'
        variant.genotypes = [[0, 1, False]]  # het
        variant.gt_depths = [60]
        variant.gt_alt_depths = [30]
        variant.gt_alt_freqs = [0.5]
        
        mock_vcf_instance.__iter__ = Mock(return_value=iter([variant]))
        
        collector = GermlineVariantCollector('dummy.vcf')
        
        # Check structure
        assert 'chr1' in collector.germline_vars
        assert 'p' in collector.germline_vars['chr1']
        assert 'hetalt' in collector.germline_vars['chr1']['p']
        assert len(collector.germline_vars['chr1']['p']['hetalt']) == 1
        
        # Check variant data
        var_data = collector.germline_vars['chr1']['p']['hetalt'][0]
        assert var_data['DP'] == 60
        assert var_data['alt_DP'] == 30
        assert var_data['VAF'] == 0.5

    @patch('germline_variant_collector.VCF')
    def test_skips_low_af_variants(self, mock_vcf_class):
        """Test variants with AF <= 0.05 are skipped."""
        mock_vcf_instance = Mock()
        mock_vcf_class.return_value = mock_vcf_instance
        mock_vcf_instance.get_header_type.return_value = {
            'Description': '"Format: gene|AF"'
        }
        
        # Variant with low AF
        variant = Mock()
        variant.CHROM = 'chr1'
        variant.POS = 100000000
        variant.INFO.get.return_value = 'BRCA1|0.01'  # AF = 0.01 < 0.05
        variant.genotypes = [[0, 1, False]]
        
        mock_vcf_instance.__iter__ = Mock(return_value=iter([variant]))
        
        collector = GermlineVariantCollector('dummy.vcf')
        
        # Should be empty - variant filtered out
        assert len(collector.germline_vars) == 0

    @patch('germline_variant_collector.VCF')
    def test_skips_variants_without_csq(self, mock_vcf_class):
        """Test variants without CSQ annotation are skipped."""
        mock_vcf_instance = Mock()
        mock_vcf_class.return_value = mock_vcf_instance
        mock_vcf_instance.get_header_type.return_value = {
            'Description': '"Format: gene|AF"'
        }
        
        variant = Mock()
        variant.INFO.get.return_value = None  # No CSQ
        
        mock_vcf_instance.__iter__ = Mock(return_value=iter([variant]))
        
        collector = GermlineVariantCollector('dummy.vcf')
        
        assert len(collector.germline_vars) == 0

    @patch('germline_variant_collector.VCF')
    def test_skips_homref_genotypes(self, mock_vcf_class):
        """Test homozygous ref variants are skipped."""
        mock_vcf_instance = Mock()
        mock_vcf_class.return_value = mock_vcf_instance
        mock_vcf_instance.get_header_type.return_value = {
            'Description': '"Format: gene|AF"'
        }
        
        variant = Mock()
        variant.CHROM = 'chr1'
        variant.POS = 100000000
        variant.INFO.get.return_value = 'BRCA1|0.1'
        variant.genotypes = [[0, 0, False]]  # homref
        
        mock_vcf_instance.__iter__ = Mock(return_value=iter([variant]))
        
        collector = GermlineVariantCollector('dummy.vcf')
        
        assert len(collector.germline_vars) == 0

    @patch('germline_variant_collector.VCF')
    def test_multiple_variants_same_arm(self, mock_vcf_class):
        """Test multiple variants accumulate correctly."""
        mock_vcf_instance = Mock()
        mock_vcf_class.return_value = mock_vcf_instance
        mock_vcf_instance.get_header_type.return_value = {
            'Description': '"Format: gene|AF"'
        }
        
        # Two het variants on chr1p
        v1 = Mock()
        v1.CHROM = 'chr1'
        v1.POS = 50000000
        v1.INFO.get.return_value = 'GENE1|0.2'
        v1.genotypes = [[0, 1, False]]
        v1.gt_depths = [40]
        v1.gt_alt_depths = [20]
        v1.gt_alt_freqs = [0.5]
        
        v2 = Mock()
        v2.CHROM = 'chr1'
        v2.POS = 60000000
        v2.INFO.get.return_value = 'GENE2|0.3'
        v2.genotypes = [[0, 1, False]]
        v2.gt_depths = [80]
        v2.gt_alt_depths = [40]
        v2.gt_alt_freqs = [0.5]
        
        mock_vcf_instance.__iter__ = Mock(return_value=iter([v1, v2]))
        
        collector = GermlineVariantCollector('dummy.vcf')
        
        assert len(collector.germline_vars['chr1']['p']['hetalt']) == 2