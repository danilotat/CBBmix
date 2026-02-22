/*
 * _baf.cpp — High-performance BAF computation from BAM.
 * Optimized for speed: blocked reference fetching, LUT decoding, threading.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <htslib/sam.h>
#include <htslib/faidx.h>
#include <htslib/hts.h>
#include <htslib/thread_pool.h>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

namespace py = pybind11;

// ─── Lookup Tables ──────────────────────────────────────────────────────────

static const uint8_t BAM_LOOKUP[16] = {
    255, 0, 1, 255, 2, 255, 255, 255, 3, 255, 255, 255, 255, 255, 255, 255
};

static int8_t CHAR_LOOKUP[256];
struct InitTables {
    InitTables() {
        std::memset(CHAR_LOOKUP, -1, 256);
        CHAR_LOOKUP['A'] = CHAR_LOOKUP['a'] = 0;
        CHAR_LOOKUP['C'] = CHAR_LOOKUP['c'] = 1;
        CHAR_LOOKUP['G'] = CHAR_LOOKUP['g'] = 2;
        CHAR_LOOKUP['T'] = CHAR_LOOKUP['t'] = 3;
    }
};
static InitTables init_tables_instance;

// ─── RAII Wrappers ──────────────────────────────────────────────────────────

struct HtsFileDeleter { void operator()(samFile *f) const { if (f) sam_close(f); } };
struct HdrDeleter { void operator()(bam_hdr_t *h) const { if (h) bam_hdr_destroy(h); } };
struct IdxDeleter { void operator()(hts_idx_t *i) const { if (i) hts_idx_destroy(i); } };
struct FaiDeleter { void operator()(faidx_t *f) const { if (f) fai_destroy(f); } };
struct ItrDeleter { void operator()(hts_itr_t *i) const { if (i) sam_itr_destroy(i); } };

using HtsFilePtr = std::unique_ptr<samFile, HtsFileDeleter>;
using HdrPtr     = std::unique_ptr<bam_hdr_t, HdrDeleter>;
using IdxPtr     = std::unique_ptr<hts_idx_t, IdxDeleter>;
using FaiPtr     = std::unique_ptr<faidx_t, FaiDeleter>;
using ItrPtr     = std::unique_ptr<hts_itr_t, ItrDeleter>;

struct PlpData {
    samFile   *fp;
    hts_itr_t *itr;
};

static int plp_callback(void *data, bam1_t *b) {
    auto *d = static_cast<PlpData *>(data);
    int ret;
    while ((ret = sam_itr_next(d->fp, d->itr, b)) >= 0) {
        if ((b->core.flag & 0xF04) == 0) return ret; 
    }
    return ret;
}

// ─── Main Function ──────────────────────────────────────────────────────────

static py::tuple compute_baf(const std::string &bam_path,
                             const std::string &ref_path,
                             const std::string &region,
                             int min_depth   = 10,
                             int min_mapq    = 20,
                             int min_baseq   = 20,
                             float min_baf   = 0.2f,
                             float max_baf   = 0.7f,
                             int min_alt     = 2,
                             const std::string &strand = "",
                             int n_threads   = 4) { 

    int strand_mode = 0;
    if (!strand.empty()) {
        if (strand == "+" || strand == "forward") strand_mode = 1;
        else if (strand == "-" || strand == "reverse") strand_mode = 2;
        else throw std::invalid_argument("Invalid strand arg");
    }

    // Thread pool must outlive the file handle (destroyed after sam_close).
    // Declare guard first so it is destroyed last.
    hts_tpool *tpool_raw = nullptr;
    struct PoolGuard {
        hts_tpool *p;
        ~PoolGuard() { if (p) hts_tpool_destroy(p); }
    } pool_guard{nullptr};

    HtsFilePtr fp(sam_open(bam_path.c_str(), "r"));
    if (!fp) throw std::runtime_error("Cannot open BAM");

    if (n_threads > 1) {
        tpool_raw = hts_tpool_init(n_threads);
        if (tpool_raw) {
            pool_guard.p = tpool_raw;
            htsThreadPool tp = {tpool_raw, 0};
            hts_set_thread_pool(fp.get(), &tp);
        }
    }

    HdrPtr hdr(sam_hdr_read(fp.get()));
    if (!hdr) throw std::runtime_error("Cannot read header");

    IdxPtr idx(sam_index_load(fp.get(), bam_path.c_str()));
    if (!idx) throw std::runtime_error("Cannot load BAM index");

    FaiPtr fai(fai_load(ref_path.c_str()));
    if (!fai) throw std::runtime_error("Cannot load Reference index");

    // Region parsing - hts_pos_t (64-bit) required here
    int r_tid = 0;
    hts_pos_t r_beg = 0, r_end = 0; 
    
    if (hts_parse_region(region.c_str(), &r_tid, &r_beg, &r_end, (hts_name2id_f)bam_name2id, hdr.get(), 0) == nullptr) {
        throw std::runtime_error("Could not parse region: " + region);
    }

    // Fetch Reference
    // faidx_fetch_seq usually expects `int`, so we cast the 64-bit coords.
    // This is safe for standard chromosomes (<2GB).
    int fetch_len_int = 0;
    char *ref_seq_ptr = faidx_fetch_seq(fai.get(), hdr->target_name[r_tid], 
                                        static_cast<int>(r_beg), 
                                        static_cast<int>(r_end - 1), 
                                        &fetch_len_int);
    
    if (!ref_seq_ptr) throw std::runtime_error("Failed to fetch reference sequence");
    std::unique_ptr<char, void(*)(void*)> ref_guard(ref_seq_ptr, std::free);
    
    size_t est_size = (r_end - r_beg) / 100; 
    if (est_size < 1000) est_size = 1000;

    std::vector<int32_t> positions;     positions.reserve(est_size);
    std::vector<int32_t> total_depths;  total_depths.reserve(est_size);
    std::vector<int32_t> alt_depths;    alt_depths.reserve(est_size);

    ItrPtr itr(sam_itr_querys(idx.get(), hdr.get(), region.c_str()));
    if (!itr) throw std::runtime_error("Cannot query region");

    PlpData pdata{fp.get(), itr.get()};
    bam_plp_t plp = bam_plp_init(plp_callback, &pdata);
    struct PlpGuard { bam_plp_t p; ~PlpGuard() { if(p) bam_plp_destroy(p); } } plp_guard_obj{plp};

    int tid, pos, n;
    const bam_pileup1_t *pile;

    while ((pile = bam_plp_auto(plp, &tid, &pos, &n)) != nullptr) {
        // Compare int pos with hts_pos_t r_beg/r_end (safe promotion)
        if (n < min_depth || tid != r_tid || pos < r_beg || pos >= r_end) continue;

        hts_pos_t ref_offset = pos - r_beg;
        if (ref_offset >= fetch_len_int) continue;

        int ref_idx = CHAR_LOOKUP[static_cast<uint8_t>(ref_seq_ptr[ref_offset])];
        if (ref_idx < 0) continue; 

        int counts[4] = {0, 0, 0, 0};
        int valid = 0;

        for (int i = 0; i < n; ++i) {
            const bam_pileup1_t &p = pile[i];
            if (p.is_del || p.is_refskip) continue;

            const bam1_t *b = p.b;
            if (b->core.qual < min_mapq) continue;

            uint32_t flag = b->core.flag;
            if (strand_mode == 1 && (flag & BAM_FREVERSE)) continue;
            if (strand_mode == 2 && !(flag & BAM_FREVERSE)) continue;

            uint8_t *qual = bam_get_qual(b);
            if (qual[p.qpos] < min_baseq) continue;

            uint8_t *seq = bam_get_seq(b);
            uint8_t bam_base = bam_seqi(seq, p.qpos);
            uint8_t idx = BAM_LOOKUP[bam_base];
            
            if (idx != 255) {
                counts[idx]++;
                valid++;
            }
        }

        if (valid < min_depth) continue;

        int max_alt_cnt = 0;
        if (ref_idx != 0 && counts[0] > max_alt_cnt) max_alt_cnt = counts[0];
        if (ref_idx != 1 && counts[1] > max_alt_cnt) max_alt_cnt = counts[1];
        if (ref_idx != 2 && counts[2] > max_alt_cnt) max_alt_cnt = counts[2];
        if (ref_idx != 3 && counts[3] > max_alt_cnt) max_alt_cnt = counts[3];

        if (max_alt_cnt < min_alt) continue;

        float baf = (float)max_alt_cnt / (float)valid;
        if (baf >= min_baf && baf <= max_baf) {
            positions.push_back(pos);
            total_depths.push_back(valid);
            alt_depths.push_back(max_alt_cnt);
        }
    }

    py::gil_scoped_acquire acquire;
    return py::make_tuple(
        py::array_t<int32_t>(positions.size(), positions.data()),
        py::array_t<int32_t>(total_depths.size(), total_depths.data()),
        py::array_t<int32_t>(alt_depths.size(), alt_depths.data())
    );
}

PYBIND11_MODULE(_baf, m) {
    m.doc() = "Optimized BAF computation";
    m.def("compute_baf", &compute_baf,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("bam_path"), py::arg("ref_path"), py::arg("region"),
          py::arg("min_depth")=10, py::arg("min_mapq")=20, py::arg("min_baseq")=20,
          py::arg("min_baf")=0.2f, py::arg("max_baf")=0.7f, py::arg("min_alt")=2,
          py::arg("strand")="", py::arg("n_threads")=4);
}