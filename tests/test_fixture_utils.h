// GGUF test fixture utilities
// Provides runtime fixture path resolution with environment variable override

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <sys/stat.h>

// Fixture metadata
typedef struct {
    const char *filename;
    const char *arch;
} marmot_test_fixture_t;

// All available GGUF fixtures
static const marmot_test_fixture_t MARMOT_TEST_FIXTURES[] = {
    {"tinyllama-q4_k_m.gguf", "llama"}, {"qwen2-0_5b-instruct-q4_k_m.gguf", "qwen2"}, {"Qwen3-0.6B-Q8_0.gguf", "qwen3"},
    {"gemma-2b.Q8_0.gguf", "gemma"},    {"Phi-3-mini-4k-instruct-q4.gguf", "phi3"},
};

#define MARMOT_TEST_FIXTURE_COUNT (sizeof(MARMOT_TEST_FIXTURES) / sizeof(MARMOT_TEST_FIXTURES[0]))

// Get fixture directory: checks MARMOT_GGUF_FIXTURE_DIR env var first,
// falls back to compile-time default
static inline const char *marmot_test_get_fixture_dir(void) {
    const char *dir = getenv("MARMOT_GGUF_FIXTURE_DIR");
    if (dir && dir[0] != '\0') {
        return dir;
    }
#ifdef MARMOT_GGUF_FIXTURE_DIR_DEFAULT
    return MARMOT_GGUF_FIXTURE_DIR_DEFAULT;
#else
    return nullptr;
#endif
}

// Build full fixture path into provided buffer
// Returns buf on success, nullptr if no fixture directory available
static inline const char *marmot_test_get_fixture_path(const char *filename, char *buf, size_t bufsize) {
    const char *dir = marmot_test_get_fixture_dir();
    if (!dir) {
        return nullptr;
    }
    snprintf(buf, bufsize, "%s/%s", dir, filename);
    return buf;
}

// Check if fixture file exists
static inline bool marmot_test_fixture_exists(const char *path) {
    if (!path) {
        return false;
    }
    struct stat st;
    return stat(path, &st) == 0;
}

// Helper macro for iterating over all fixtures
#define MARMOT_TEST_FOR_EACH_FIXTURE(fixture_var)                                                                      \
    for (size_t _fixture_idx = 0; _fixture_idx < MARMOT_TEST_FIXTURE_COUNT; _fixture_idx++)                            \
        if ((fixture_var = &MARMOT_TEST_FIXTURES[_fixture_idx]))

// Maximum path buffer size
#define MARMOT_TEST_PATH_MAX 4096
