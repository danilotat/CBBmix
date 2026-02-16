"""Tests for the BAF extension module."""

from unittest.mock import patch

import pytest

from CBBmix.baf import HAS_BAF_EXTENSION, compute_baf, compute_baf_genome


class TestImportAvailability:
    """Verify the extension can be imported (or gracefully skipped)."""

    def test_has_baf_extension_is_bool(self):
        assert isinstance(HAS_BAF_EXTENSION, bool)

    @pytest.mark.skipif(not HAS_BAF_EXTENSION, reason="C++ extension not compiled")
    def test_native_module_importable(self):
        from CBBmix._baf import compute_baf as _native

        assert callable(_native)


class TestErrorHandling:
    """Verify proper errors on invalid inputs."""

    @pytest.mark.skipif(not HAS_BAF_EXTENSION, reason="C++ extension not compiled")
    def test_bad_bam_path(self):
        with pytest.raises(RuntimeError, match="Cannot open BAM"):
            compute_baf("/nonexistent.bam", "/nonexistent.fa", "chr1")

    @pytest.mark.skipif(not HAS_BAF_EXTENSION, reason="C++ extension not compiled")
    def test_bad_ref_path(self, tmp_path):
        # create a dummy file so htslib can attempt to open it
        dummy_bam = tmp_path / "dummy.bam"
        dummy_bam.touch()
        with pytest.raises(RuntimeError):
            compute_baf(str(dummy_bam), "/nonexistent.fa", "chr1")


class TestFallbackLogic:
    """Verify behaviour when the C++ extension is absent."""

    def test_compute_baf_raises_without_extension(self):
        with patch("CBBmix.baf.HAS_BAF_EXTENSION", False):
            with pytest.raises(RuntimeError, match="BAF extension not available"):
                compute_baf("x.bam", "x.fa", "chr1")

    def test_compute_baf_genome_raises_without_extension(self):
        with patch("CBBmix.baf.HAS_BAF_EXTENSION", False):
            with pytest.raises(RuntimeError, match="BAF extension not available"):
                compute_baf_genome("x.bam", "x.fa")


class TestGenomeConvenience:
    """Test compute_baf_genome helper logic."""

    def test_default_chromosomes(self):
        from CBBmix.baf import CHROMOSOMES

        assert len(CHROMOSOMES) == 24
        assert CHROMOSOMES[0] == "chr1"
        assert CHROMOSOMES[-1] == "chrY"
