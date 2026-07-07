// CPU-only unit test for the level-2 (disk) prompt cache tier: server_prompt_disk_cache.
//
// Exercises blob (de)serialization + CRC round-trip, the in-memory prefix index, LRU size eviction,
// persistence/rescan across "restarts", and corruption handling. Requires no GPU and no llama context.

#include "server-task.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static int g_fail = 0;
#define CHECK(cond) do { \
        if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } \
        else         { fprintf(stderr, "ok   %s\n", #cond); } \
    } while (0)

static server_prompt make_prompt(const llama_tokens & toks, uint8_t fill, size_t main_bytes, size_t drft_bytes = 0) {
    server_prompt p;
    p.tokens = server_tokens(toks, false);
    p.data.main.assign(main_bytes, fill);
    p.data.drft.assign(drft_bytes, (uint8_t)(fill ^ 0xFF));
    return p;
}

// wait until the async writer has flushed `n` entries to the index (or timeout)
static bool wait_for_index(server_prompt_disk_cache & dc, size_t n, int timeout_ms = 5000) {
    const auto t0 = std::chrono::steady_clock::now();
    while (dc.index_size() != n) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

// wait until a prefix-matching entry for `toks` is (present==true) / is not (present==false) on disk
static bool wait_for_present(server_prompt_disk_cache & dc, const llama_tokens & toks, bool present, int timeout_ms = 5000) {
    const auto t0 = std::chrono::steady_clock::now();
    server_tokens q(toks, false);
    int lcp; float fk, sim;
    while (dc.find_best(q, 0.25f, lcp, fk, sim).empty() == present) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

int main() {
    const auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string dir = (fs::temp_directory_path() / ("kvdisk_test_" + std::to_string((long long) now_ns))).string();
    fs::remove_all(dir);

    // ---------------------------------------------------------------------
    // 1. write + prefix lookup + blob round-trip
    // ---------------------------------------------------------------------
    {
        server_prompt_disk_cache dc(dir, /*limit_mib*/ 0 /* no limit */);
        dc.init();
        CHECK(dc.index_size() == 0);

        dc.offload(make_prompt({1,2,3,4,5,6,7,8},        0xAA, 4096));
        dc.offload(make_prompt({1,2,3,4,9,9,9,9,9,9},    0xBB, 8192, 1024));
        CHECK(wait_for_index(dc, 2));
        CHECK(dc.index_bytes() > 0);

        // a query sharing an 8-token prefix with entry #1 must select it (higher f_keep + sim)
        server_tokens q1(llama_tokens{1,2,3,4,5,6,7,8,100,101}, false);
        int lcp; float fk, sim;
        std::string best = dc.find_best(q1, 0.25f, lcp, fk, sim);
        CHECK(!best.empty());
        CHECK(lcp == 8);

        // round-trip that entry
        server_prompt out;
        CHECK(dc.load_blob(best, out));
        CHECK(out.tokens.get_tokens() == (llama_tokens{1,2,3,4,5,6,7,8}));
        CHECK(out.data.main.size() == 4096);
        CHECK(out.data.main[0] == 0xAA && out.data.main[4095] == 0xAA);
        // consumed on load
        CHECK(wait_for_index(dc, 1, 1000));

        // a query that shares only a 4/10 = 0.4 f_keep... entry #2 tokens len 10, lcp 4 -> f_keep 0.4 (>0.25) ok
        server_tokens q2(llama_tokens{1,2,3,4}, false);
        best = dc.find_best(q2, 0.25f, lcp, fk, sim);
        CHECK(!best.empty());
        CHECK(lcp == 4);

        // below-threshold query: only 1 shared token out of a 10-token entry -> f_keep 0.1 < 0.25 -> miss
        server_tokens q3(llama_tokens{1,500,500}, false);
        best = dc.find_best(q3, 0.25f, lcp, fk, sim);
        CHECK(best.empty());
    }

    // ---------------------------------------------------------------------
    // 2. persistence across "restart" (rescan of the directory)
    // ---------------------------------------------------------------------
    {
        server_prompt_disk_cache dc2(dir, 0);
        dc2.init();
        // entry #2 (8192+1024 bytes) survived from phase 1
        CHECK(dc2.index_size() == 1);
        server_tokens q(llama_tokens{1,2,3,4,9,9}, false);
        int lcp; float fk, sim;
        std::string best = dc2.find_best(q, 0.25f, lcp, fk, sim);
        CHECK(!best.empty());
        server_prompt out;
        CHECK(dc2.load_blob(best, out));
        CHECK(out.data.main.size() == 8192);
        CHECK(out.data.drft.size() == 1024);
        CHECK(out.data.drft[0] == (uint8_t)(0xBB ^ 0xFF));
    }

    // ---------------------------------------------------------------------
    // 3. LRU size eviction
    // ---------------------------------------------------------------------
    {
        fs::remove_all(dir);
        // limit ~0.75 MiB; each blob is >0.5 MiB, so only one fits at a time -> every new write evicts the LRU
        server_prompt_disk_cache dc(dir, 1);
        dc.init();

        // write #1, wait for it to land
        dc.offload(make_prompt({10,11,12,13}, 0x11, 640*1024));
        CHECK(wait_for_present(dc, {10,11,12,13}, true));
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // ensure distinct mtimes

        // write #2 exceeds the 1 MiB limit (2 x >0.625 MiB) -> LRU (#1) evicted
        dc.offload(make_prompt({20,21,22,23}, 0x22, 640*1024));
        CHECK(wait_for_present(dc, {20,21,22,23}, true));  // #2 landed
        CHECK(wait_for_present(dc, {10,11,12,13}, false)); // #1 evicted as LRU
        CHECK(dc.index_bytes() <= 1024*1024);

        // write #3 -> #2 evicted
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        dc.offload(make_prompt({30,31,32,33}, 0x33, 640*1024));
        CHECK(wait_for_present(dc, {30,31,32,33}, true));
        CHECK(wait_for_present(dc, {20,21,22,23}, false));
        CHECK(dc.index_size() == 1);
    }

    // ---------------------------------------------------------------------
    // 4. corruption handling: a truncated/garbled blob must fail load and be purged
    // ---------------------------------------------------------------------
    {
        fs::remove_all(dir);
        server_prompt_disk_cache dc(dir, 0);
        dc.init();
        dc.offload(make_prompt({7,7,7,7,7,7}, 0x55, 4096));
        CHECK(wait_for_index(dc, 1));

        // locate the blob file and corrupt its middle
        std::string blob;
        for (const auto & de : fs::directory_iterator(dir)) {
            if (de.path().extension() == ".kvblob") { blob = de.path().string(); break; }
        }
        CHECK(!blob.empty());
        {
            std::fstream f(blob, std::ios::binary | std::ios::in | std::ios::out);
            f.seekp(2048);
            char junk[64]; for (auto & c : junk) c = 0x7E;
            f.write(junk, sizeof(junk));
        }

        server_tokens q(llama_tokens{7,7,7,7,7,7}, false);
        int lcp; float fk, sim;
        std::string best = dc.find_best(q, 0.25f, lcp, fk, sim);
        CHECK(!best.empty());
        server_prompt out;
        CHECK(!dc.load_blob(best, out));                 // CRC mismatch -> false
        CHECK(wait_for_index(dc, 0, 1000));              // bad entry purged
        CHECK(!fs::exists(blob));                         // file deleted
    }

    fs::remove_all(dir);

    if (g_fail == 0) {
        fprintf(stderr, "\nALL DISK-CACHE TESTS PASSED\n");
        return 0;
    }
    fprintf(stderr, "\n%d CHECK(S) FAILED\n", g_fail);
    return 1;
}
