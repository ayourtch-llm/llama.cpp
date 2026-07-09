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

    // ---------------------------------------------------------------------
    // 5. non-destructive load: a valid blob read with consume=false must survive on disk
    //    (models the restore path that reads the blob but whose set_data may still fail for
    //    lack of free KV cells); an explicit consume() then removes it.
    // ---------------------------------------------------------------------
    {
        fs::remove_all(dir);
        server_prompt_disk_cache dc(dir, 0);
        dc.init();
        dc.offload(make_prompt({4,4,4,4,4,4}, 0x33, 4096));
        CHECK(wait_for_index(dc, 1));

        server_tokens q(llama_tokens{4,4,4,4,4,4}, false);
        int lcp; float fk, sim;
        std::string best = dc.find_best(q, 0.25f, lcp, fk, sim);
        CHECK(!best.empty());

        // read non-destructively: succeeds, but the entry must remain (simulates a restore that
        // fails afterwards and needs to reload the blob on the swap-in retry).
        server_prompt out;
        CHECK(dc.load_blob(best, out, /*consume=*/false));
        CHECK(out.data.main.size() == 4096);
        CHECK(dc.index_size() == 1);                      // NOT consumed
        CHECK(fs::exists(best));                           // file still present

        // a second read of the same blob still works (retry path)
        server_prompt out2;
        CHECK(dc.load_blob(best, out2, /*consume=*/false));
        CHECK(out2.data.main.size() == 4096);
        CHECK(dc.index_size() == 1);

        // explicit consume (restore committed) removes index entry + file exactly once
        dc.consume(best);
        CHECK(dc.index_size() == 0);
        CHECK(!fs::exists(best));
        dc.consume(best);                                 // idempotent no-op when already gone
        CHECK(dc.index_size() == 0);
    }

    // ---------------------------------------------------------------------
    // 6. drain() barrier: after offloading several entries, drain() must block until every blob is
    //    on disk (proves the graceful-shutdown flush guarantee).
    // ---------------------------------------------------------------------
    {
        fs::remove_all(dir);
        server_prompt_disk_cache dc(dir, 0);
        dc.init();

        const size_t N = 8;
        for (size_t i = 0; i < N; i++) {
            // distinct token vectors + sizeable blobs so writes take real (if small) time
            llama_tokens toks = {1000 + (llama_token) i, 2000 + (llama_token) i, 3000 + (llama_token) i};
            dc.offload(make_prompt(toks, (uint8_t)(0x40 + i), 256 * 1024));
        }

        // drain must not return until the writer has emptied the queue and finished the in-flight write
        dc.drain();

        // immediately after drain() (no polling): everything is indexed and each blob is present
        CHECK(dc.index_size() == N);
        for (size_t i = 0; i < N; i++) {
            llama_tokens toks = {1000 + (llama_token) i, 2000 + (llama_token) i, 3000 + (llama_token) i};
            server_tokens q(toks, false);
            int lcp; float fk, sim;
            CHECK(!dc.find_best(q, 0.25f, lcp, fk, sim).empty());
        }

        // draining an already-empty queue is a cheap no-op that returns promptly
        dc.drain();
        CHECK(dc.index_size() == N);
    }

    // =====================================================================
    // --cache-shared: coherent multi-process sharing of one --cache-disk dir
    // =====================================================================

    // ---------------------------------------------------------------------
    // T-shared-1: model-namespaced filenames. Two caches with different model_ns over one dir each
    // see ONLY their own entries; no cross-restore.
    // ---------------------------------------------------------------------
    {
        fs::remove_all(dir);
        server_prompt_disk_cache a(dir, 0, "/models/A.gguf", /*geom*/ 111, /*shared*/ true);
        server_prompt_disk_cache b(dir, 0, "/models/B.gguf", /*geom*/ 222, /*shared*/ true);
        a.init();
        b.init();

        a.offload(make_prompt({1,2,3,4,5,6}, 0xAA, 4096));
        CHECK(wait_for_index(a, 1));

        server_tokens q(llama_tokens{1,2,3,4,5,6}, false);
        int lcp; float fk, sim;

        // b (different model_ns) must not see or restore a's blob, even though find_best rescans the dir
        CHECK(b.find_best(q, 0.25f, lcp, fk, sim).empty());
        CHECK(b.index_size() == 0);

        // a sees its own entry
        std::string best_a = a.find_best(q, 0.25f, lcp, fk, sim);
        CHECK(!best_a.empty());
        CHECK(a.index_size() == 1);
    }

    // ---------------------------------------------------------------------
    // T-shared-2: runtime index refresh. cacheA writes X; cacheB (already inited, SAME model_ns/geom,
    // same dir) initially misses, then a dir-mtime-triggered rescan inside find_best surfaces X.
    // Assert both an exact hit and a prefix (LCP) hit.
    // ---------------------------------------------------------------------
    {
        fs::remove_all(dir);
        server_prompt_disk_cache A(dir, 0, "/m/same.gguf", 7, true);
        server_prompt_disk_cache B(dir, 0, "/m/same.gguf", 7, true);
        A.init();
        B.init();
        CHECK(B.index_size() == 0);   // B booted against the empty dir

        A.offload(make_prompt({10,11,12,13,14,15,16,17}, 0xCD, 8192));
        CHECK(wait_for_index(A, 1));
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // ensure the dir mtime is observably newer

        int lcp; float fk, sim;

        // prefix / LCP hit: a 10-token query sharing an 8-token prefix triggers B's rescan and matches
        server_tokens qx(llama_tokens{10,11,12,13,14,15,16,17,99,99}, false);
        std::string best = B.find_best(qx, 0.25f, lcp, fk, sim);
        CHECK(!best.empty());
        CHECK(lcp == 8);
        CHECK(B.index_size() == 1);   // rescan picked up the peer's write

        // exact hit
        server_tokens qe(llama_tokens{10,11,12,13,14,15,16,17}, false);
        CHECK(!B.find_best(qe, 0.25f, lcp, fk, sim).empty());

        // and B can actually restore it (non-consuming, since shared)
        server_prompt out;
        CHECK(B.load_blob(best, out, /*consume=*/false));
        CHECK(out.data.main.size() == 8192);
    }

    // ---------------------------------------------------------------------
    // T-shared-3: coherent cap. Two shared caches, one shared dir + one 1 MiB limit; interleaved
    // writes must keep the TRUE on-disk total <= limit (not ~2x), with the global-mtime-oldest as victim.
    // ---------------------------------------------------------------------
    {
        fs::remove_all(dir);
        server_prompt_disk_cache A(dir, 1, "/m/s.gguf", 5, true);   // 1 MiB shared cap
        server_prompt_disk_cache B(dir, 1, "/m/s.gguf", 5, true);
        A.init();
        B.init();

        A.offload(make_prompt({1,1,1,1}, 0x11, 640*1024));
        CHECK(wait_for_present(A, {1,1,1,1}, true));
        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // A's blob is strictly older

        // B's write pushes 2 x 640 KiB > 1 MiB; B's coherent evict (flock + rescan) drops A's older blob
        B.offload(make_prompt({2,2,2,2}, 0x22, 640*1024));
        CHECK(wait_for_present(B, {2,2,2,2}, true));
        CHECK(wait_for_present(B, {1,1,1,1}, false));   // global-mtime LRU victim is A's blob

        // true on-disk total (independent of any single index) stays under the shared cap
        size_t on_disk = 0;
        for (const auto & de : fs::directory_iterator(dir)) {
            if (de.path().extension() == ".kvblob") on_disk += fs::file_size(de.path());
        }
        CHECK(on_disk <= 1024*1024);
    }

    // ---------------------------------------------------------------------
    // T-shared-4: evict race under the cross-process lock. Two shared caches hammer one capped dir;
    // no crash / torn index, the cap holds, and remove is idempotent.
    // ---------------------------------------------------------------------
    {
        fs::remove_all(dir);
        server_prompt_disk_cache A(dir, 1, "/m/r.gguf", 9, true);
        server_prompt_disk_cache B(dir, 1, "/m/r.gguf", 9, true);
        A.init();
        B.init();

        for (int i = 0; i < 6; i++) {
            A.offload(make_prompt({(llama_token)(100+i)}, (uint8_t)(0x30+i), 400*1024));
            B.offload(make_prompt({(llama_token)(200+i)}, (uint8_t)(0x50+i), 400*1024));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        A.drain();
        B.drain();

        // the last coherent evict (under flock) saw every renamed blob -> total is under the shared cap
        size_t on_disk = 0; size_t n_blobs = 0;
        for (const auto & de : fs::directory_iterator(dir)) {
            if (de.path().extension() == ".kvblob") { on_disk += fs::file_size(de.path()); n_blobs++; }
        }
        CHECK(on_disk <= 1024*1024);
        CHECK(n_blobs <= 2);

        // a rescan-driven query still works (index not torn), and consuming a vanished path is a no-op
        int lcp; float fk, sim;
        server_tokens q(llama_tokens{100}, false);
        (void) A.find_best(q, 0.25f, lcp, fk, sim);         // must not crash
        A.consume(dir + "/deadbeef.kvblob");                 // idempotent: no entry, no crash
        A.consume(dir + "/deadbeef.kvblob");
        CHECK(true);                                          // reaching here = no double-free / crash
    }

    // ---------------------------------------------------------------------
    // T-shared-5: non-consuming restore. A shared-mode restore leaves the blob on disk (touch, not
    // consume) so a peer can still reload it.
    // ---------------------------------------------------------------------
    {
        fs::remove_all(dir);
        server_prompt_disk_cache A(dir, 0, "/m/n.gguf", 3, true);
        server_prompt_disk_cache B(dir, 0, "/m/n.gguf", 3, true);
        A.init();
        B.init();

        A.offload(make_prompt({5,5,5,5,5,5}, 0x77, 4096));
        CHECK(wait_for_index(A, 1));

        int lcp; float fk, sim;
        server_tokens q(llama_tokens{5,5,5,5,5,5}, false);
        std::string best = A.find_best(q, 0.25f, lcp, fk, sim);
        CHECK(!best.empty());

        // shared restore = load_blob(consume=false) + touch() (what server_prompt_cache::load does)
        server_prompt out;
        CHECK(A.load_blob(best, out, /*consume=*/false));
        A.touch(best);
        CHECK(A.index_size() == 1);      // NOT consumed
        CHECK(fs::exists(best));         // blob left on disk for peers

        // peer B rescans, finds the same blob and reloads it
        std::string best_b = B.find_best(q, 0.25f, lcp, fk, sim);
        CHECK(!best_b.empty());
        server_prompt out2;
        CHECK(B.load_blob(best_b, out2, /*consume=*/false));
        CHECK(out2.data.main.size() == 4096);
        CHECK(fs::exists(best_b));
    }

    // ---------------------------------------------------------------------
    // T-shared-6 (geometry fingerprint): a blob written under one geometry is rejected (rescan-skipped
    // AND load-rejected) by a same-model_ns instance with a different geometry, and is NOT purged.
    // ---------------------------------------------------------------------
    {
        fs::remove_all(dir);
        server_prompt_disk_cache w(dir, 0, "/m/g.gguf", /*geom*/ 1000, false);
        w.init();
        w.offload(make_prompt({8,8,8,8}, 0x12, 4096));
        CHECK(wait_for_index(w, 1));

        int lcp; float fk, sim;
        server_tokens q(llama_tokens{8,8,8,8}, false);
        std::string best = w.find_best(q, 0.25f, lcp, fk, sim);
        CHECK(!best.empty());

        // same model_ns (same path), DIFFERENT geometry: rescan drops it, direct load rejects it
        server_prompt_disk_cache r(dir, 0, "/m/g.gguf", /*geom*/ 2000, true);
        r.init();
        CHECK(r.index_size() == 0);
        CHECK(r.find_best(q, 0.25f, lcp, fk, sim).empty());

        server_prompt out;
        CHECK(!r.load_blob(best, out, /*consume=*/false));   // geometry mismatch -> false
        CHECK(fs::exists(best));                              // not corrupt -> not purged
    }

    fs::remove_all(dir);

    if (g_fail == 0) {
        fprintf(stderr, "\nALL DISK-CACHE TESTS PASSED\n");
        return 0;
    }
    fprintf(stderr, "\n%d CHECK(S) FAILED\n", g_fail);
    return 1;
}
