#ifndef MARMOT_TOKENIZER_H
#define MARMOT_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "error.h"
#include "macros.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MARMOT_TOKENIZER_OPTIONS_VERSION 1
#define MARMOT_TOKENIZER_ENCODE_OPTIONS_VERSION 1
#define MARMOT_TOKENIZER_DECODE_OPTIONS_VERSION 1

typedef int32_t marmot_token_id_t;
#define MARMOT_TOKEN_ID_INVALID ((marmot_token_id_t) - 1)

typedef enum {
    MARMOT_TOKENIZER_FLAG_STRICT_VALIDATION = 1u << 0,
    MARMOT_TOKENIZER_FLAG_ENABLE_CACHE = 1u << 1,
} marmot_tokenizer_flags_t;

typedef struct marmot_tokenizer marmot_tokenizer_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;
    const void *pnext;
    uint64_t reserved[4];
} marmot_tokenizer_options_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    bool add_bos;
    bool add_eos;
    bool allow_special;
    size_t max_tokens;
    const void *pnext;
    uint64_t reserved[4];
} marmot_tokenizer_encode_options_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    bool skip_special;
    bool strip_space_prefix;
    const void *pnext;
    uint64_t reserved[4];
} marmot_tokenizer_decode_options_t;

typedef struct {
    bool has_bos;
    bool has_eos;
    bool has_unk;
    bool has_pad;
    marmot_token_id_t bos_id;
    marmot_token_id_t eos_id;
    marmot_token_id_t unk_id;
    marmot_token_id_t pad_id;
} marmot_tokenizer_special_ids_t;

typedef struct {
    bool has_add_bos;
    bool add_bos;
    bool has_add_eos;
    bool add_eos;
} marmot_tokenizer_behavior_t;

MARMOT_NODISCARD marmot_error_t marmot_tokenizer_options_init(marmot_tokenizer_options_t *opts);
MARMOT_NODISCARD marmot_error_t marmot_tokenizer_encode_options_init(marmot_tokenizer_encode_options_t *opts);
MARMOT_NODISCARD marmot_error_t marmot_tokenizer_decode_options_init(marmot_tokenizer_decode_options_t *opts);

MARMOT_NODISCARD marmot_error_t marmot_tokenizer_create_from_gguf_file(
    const char *path, const marmot_tokenizer_options_t *opts, marmot_tokenizer_t **out_tokenizer
);
void marmot_tokenizer_destroy(marmot_tokenizer_t *tokenizer);

MARMOT_NODISCARD size_t marmot_tokenizer_vocab_size(const marmot_tokenizer_t *tokenizer);
MARMOT_NODISCARD marmot_error_t
marmot_tokenizer_get_special_ids(const marmot_tokenizer_t *tokenizer, marmot_tokenizer_special_ids_t *out_special_ids);
MARMOT_NODISCARD marmot_error_t
marmot_tokenizer_get_behavior(const marmot_tokenizer_t *tokenizer, marmot_tokenizer_behavior_t *out_behavior);

MARMOT_NODISCARD marmot_error_t marmot_tokenizer_piece_to_token(
    const marmot_tokenizer_t *tokenizer, const char *piece, size_t piece_len, marmot_token_id_t *out_token_id
);

MARMOT_NODISCARD marmot_error_t marmot_tokenizer_token_to_piece(
    const marmot_tokenizer_t *tokenizer, marmot_token_id_t token_id, char *out_piece, size_t *inout_len
);

MARMOT_NODISCARD marmot_error_t
marmot_tokenizer_chat_template(const marmot_tokenizer_t *tokenizer, char *out_template, size_t *inout_len);

MARMOT_NODISCARD marmot_error_t marmot_tokenizer_encode(
    const marmot_tokenizer_t *tokenizer, const char *text, size_t text_len,
    const marmot_tokenizer_encode_options_t *opts, marmot_token_id_t *out_token_ids, size_t *inout_len
);

MARMOT_NODISCARD marmot_error_t marmot_tokenizer_decode(
    const marmot_tokenizer_t *tokenizer, const marmot_token_id_t *token_ids, size_t token_ids_len,
    const marmot_tokenizer_decode_options_t *opts, char *out_text, size_t *inout_len
);

MARMOT_NODISCARD const marmot_error_info_t *marmot_tokenizer_last_error(const marmot_tokenizer_t *tokenizer);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_TOKENIZER_H
