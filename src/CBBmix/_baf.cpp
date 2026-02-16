/*
 * _baf.cpp — High-performance BAF computation from BAM via htslib pileup.
 *
 * Returns (positions, bafs) as NumPy arrays for a given chromosome region.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <htslib/sam.h>
#include <htslib/faidx.h>
#include <htslib/hts.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ─── RAII wrappers ──────────────────────────────────────────────────────────

struct HtsFileDeleter {
    void operator()(samFile *f) const { if (f) sam_close(f); }
};
struct HdrDeleter {
    void operator()(bam_hdr_t *h) const { if (h) bam_hdr_destroy(h); }
};
struct IdxDeleter {
    void operator()(hts_idx_t *i) const { if (i) hts_idx_destroy(i); }
};
struct FaiDeleter {
    void operator()(faidx_t *f) const { if (f) fai_destroy(f); }
};
struct ItrDeleter {
    void operator()(hts_itr_t *i) const { if (i) sam_itr_destroy(i); }
};
using HtsFilePtr = std::unique_ptr<samFile, HtsFileDeleter>;
using HdrPtr     = std::unique_ptr<bam_hdr_t, HdrDeleter>;
using IdxPtr     = std::unique_ptr<hts_idx_t, IdxDeleter>;
using FaiPtr     = std::unique_ptr<faidx_t, FaiDeleter>;
using ItrPtr     = std::unique_ptr<hts_itr_t, ItrDeleter>;

// ─── Pileup callback ───────────────────────────────────────────────────────

struct PlpData {
    samFile   *fp;
    bam_hdr_t *hdr;
    hts_itr_t *itr;
};

static int plp_callback(void *data, bam1_t *b) {
    auto *d = static_cast<PlpData *>(data);
    int ret;
    while (true) {
        ret = sam_itr_next(d->fp, d->itr, b);
        if (ret < 0) return ret;
        // skip unmapped, secondary, supplementary, duplicate, failed-QC
        if (b->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY |
                            BAM_FDUP | BAM_FQCFAIL))
            continue;
        return ret;
    }
}

// ─── Base helpers ───────────────────────────────────────────────────────────

static inline int base_to_idx(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

static const char SEQ_NT16[] = "=ACMGRSVTWYHKDBN";

// ─── Main function ──────────────────────────────────────────────────────────

static py::tuple compute_baf(const std::string &bam_path,
                             const std::string &ref_path,
                             const std::string &region,
                             int min_depth   = 10,
                             int min_mapq    = 20,
                             int min_baseq   = 20,
                             float min_baf   = 0.2f,
                             float max_baf   = 0.7f) {
    // Open BAM
    HtsFilePtr fp(sam_open(bam_path.c_str(), "r"));
    if (!fp) throw std::runtime_error("Cannot open BAM: " + bam_path);

    // Read header
    HdrPtr hdr(sam_hdr_read(fp.get()));
    if (!hdr) throw std::runtime_error("Cannot read BAM header: " + bam_path);

    // Load BAM index
    IdxPtr idx(sam_index_load(fp.get(), bam_path.c_str()));
    if (!idx) throw std::runtime_error("Cannot load BAM index for: " + bam_path);

    // Load reference FASTA
    FaiPtr fai(fai_load(ref_path.c_str()));
    if (!fai) throw std::runtime_error("Cannot load reference FASTA: " + ref_path);

    // Create region iterator
    ItrPtr itr(sam_itr_querys(idx.get(), hdr.get(), region.c_str()));
    if (!itr) throw std::runtime_error("Cannot query region: " + region);

    // Setup pileup with RAII guard
    PlpData pdata{fp.get(), hdr.get(), itr.get()};
    bam_plp_t plp = bam_plp_init(plp_callback, &pdata);
    if (!plp) throw std::runtime_error("Cannot init pileup engine");
    struct PlpGuard {
        bam_plp_t p;
        ~PlpGuard() { if (p) bam_plp_destroy(p); }
    } plp_guard{plp};

    // Result vectors
    std::vector<int32_t> positions;
    std::vector<float>   bafs;

    int tid, pos, n;
    const bam_pileup1_t *pile;

    while ((pile = bam_plp_auto(plp, &tid, &pos, &n)) != nullptr) {
        if (n < min_depth) continue;

        // Fetch reference base
        int ref_len = 0;
        char *ref_seq = faidx_fetch_seq(fai.get(), hdr->target_name[tid],
                                        pos, pos, &ref_len);
        if (!ref_seq || ref_len < 1) {
            free(ref_seq);
            continue;
        }
        int ref_idx = base_to_idx(ref_seq[0]);
        free(ref_seq);
        if (ref_idx < 0) continue;

        // Tally ACGT
        int counts[4] = {0, 0, 0, 0};
        int valid = 0;

        for (int i = 0; i < n; ++i) {
            if (pile[i].is_del || pile[i].is_refskip) continue;
            const bam1_t *b = pile[i].b;
            if (static_cast<int>(b->core.qual) < min_mapq) continue;

            uint8_t *qual = bam_get_qual(b);
            if (static_cast<int>(qual[pile[i].qpos]) < min_baseq) continue;

            uint8_t *seq = bam_get_seq(b);
            int base = bam_seqi(seq, pile[i].qpos);
            char c = SEQ_NT16[base];
            int idx = base_to_idx(c);
            if (idx >= 0) {
                counts[idx]++;
                valid++;
            }
        }

        if (valid < min_depth) continue;

        // Find max alternate allele count
        int max_alt = 0;
        for (int i = 0; i < 4; ++i) {
            if (i != ref_idx && counts[i] > max_alt) {
                max_alt = counts[i];
            }
        }

        float baf = static_cast<float>(max_alt) / static_cast<float>(valid);
        if (baf >= min_baf && baf <= max_baf) {
            positions.push_back(pos);
            bafs.push_back(baf);
        }
    }

    // Reacquire GIL before creating Python objects (call_guard released it)
    py::gil_scoped_acquire acquire;

    // Convert to NumPy arrays
    py::array_t<int32_t> pos_arr(static_cast<py::ssize_t>(positions.size()),
                                  positions.data());
    py::array_t<float>   baf_arr(static_cast<py::ssize_t>(bafs.size()),
                                  bafs.data());

    return py::make_tuple(pos_arr, baf_arr);
}

// ─── Pybind11 module ────────────────────────────────────────────────────────

PYBIND11_MODULE(_baf, m) {
    m.doc() = "High-performance BAF computation from BAM files using htslib pileup";

    m.def("compute_baf", &compute_baf,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("bam_path"),
          py::arg("ref_path"),
          py::arg("region"),
          py::arg("min_depth")  = 10,
          py::arg("min_mapq")   = 20,
          py::arg("min_baseq")  = 20,
          py::arg("min_baf")    = 0.2f,
          py::arg("max_baf")    = 0.7f,
          R"doc(
Compute B-Allele Frequencies from a BAM file for a genomic region.

Parameters
----------
bam_path : str
    Path to indexed BAM file.
ref_path : str
    Path to reference FASTA (with .fai index).
region : str
    Genomic region, e.g. "chr1" or "chr1:1000000-2000000".
min_depth : int
    Minimum read depth to consider a position (default: 10).
min_mapq : int
    Minimum mapping quality (default: 20).
min_baseq : int
    Minimum base quality (default: 20).
min_baf : float
    Minimum BAF to report (default: 0.1).
max_baf : float
    Maximum BAF to report (default: 1.0).

Returns
-------
tuple[np.ndarray, np.ndarray]
    (positions, bafs) — int32 positions and float32 BAF values.
)doc");
}
