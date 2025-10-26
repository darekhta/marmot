#ifndef MARMOT_ERROR_H
#define MARMOT_ERROR_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

MARMOT_NODISCARD const char *marmot_error_string(marmot_error_t error);

// Enhanced error info with source location (Phase 4)
typedef struct marmot_error_info {
    marmot_error_t code;
    char message[256];
    const char *file;
    int line;
    const char *function;
} marmot_error_info_t;

extern thread_local marmot_error_t marmot_last_error;
extern thread_local char marmot_last_error_detail[256];
extern thread_local marmot_error_info_t marmot_last_error_info;

// Basic error handling (legacy)
MARMOT_NODISCARD marmot_error_t marmot_get_last_error(void);
MARMOT_NODISCARD const char *marmot_get_last_error_detail(void);
void marmot_set_error(marmot_error_t error, const char *detail);
void marmot_clear_error(void);

// Enhanced error handling with source location (Phase 4)
void marmot_set_error_ex(marmot_error_t error, const char *detail, const char *file, int line, const char *function);
MARMOT_NODISCARD const marmot_error_info_t *marmot_get_last_error_info(void);

// Convenience macro for setting errors with source location
#define MARMOT_SET_ERROR(code, msg) marmot_set_error_ex(code, msg, __FILE__, __LINE__, __func__)

// Convenience macro for returning errors with source location
#define MARMOT_RETURN_ERROR(code, msg)                                                                                 \
    do {                                                                                                               \
        MARMOT_SET_ERROR(code, msg);                                                                                   \
        return code;                                                                                                   \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif
