#include "marmot/error.h"

#include <stdio.h>

#include <string.h>

thread_local marmot_error_t marmot_last_error;
thread_local char marmot_last_error_detail[256];
thread_local marmot_error_info_t marmot_last_error_info = {0};

const char *marmot_error_string(marmot_error_t error) {
    switch (error) {
    case MARMOT_SUCCESS:
        return "Success";
    case MARMOT_ERROR_INVALID_ARGUMENT:
        return "Invalid argument";
    case MARMOT_ERROR_OUT_OF_MEMORY:
        return "Out of memory";
    case MARMOT_ERROR_DEVICE_NOT_AVAILABLE:
        return "Device not available";
    case MARMOT_ERROR_BACKEND_INIT_FAILED:
        return "Backend initialization failed";
    case MARMOT_ERROR_INVALID_OPERATION:
        return "Invalid operation";
    case MARMOT_ERROR_UNSUPPORTED_DTYPE:
        return "Unsupported data type";
    case MARMOT_ERROR_DIMENSION_MISMATCH:
        return "Dimension mismatch";
    case MARMOT_ERROR_NOT_IMPLEMENTED:
        return "Not implemented";
    default:
        return "Unknown error";
    }
}

marmot_error_t marmot_get_last_error(void) {
    return marmot_last_error;
}

const char *marmot_get_last_error_detail(void) {
    return marmot_last_error_detail;
}

void marmot_set_error(marmot_error_t error, const char *detail) {
    marmot_last_error = error;

    marmot_last_error_info.code = error;
    marmot_last_error_info.file = nullptr;
    marmot_last_error_info.line = 0;
    marmot_last_error_info.function = nullptr;

    if (detail != nullptr) {
        snprintf(marmot_last_error_detail, sizeof(marmot_last_error_detail), "%s", detail);
        snprintf(marmot_last_error_info.message, sizeof(marmot_last_error_info.message), "%s", detail);
    } else {
        marmot_last_error_detail[0] = '\0';
        marmot_last_error_info.message[0] = '\0';
    }
}

void marmot_clear_error(void) {
    marmot_last_error = MARMOT_SUCCESS;
    marmot_last_error_detail[0] = '\0';
    marmot_last_error_info.code = MARMOT_SUCCESS;
    marmot_last_error_info.message[0] = '\0';
    marmot_last_error_info.file = nullptr;
    marmot_last_error_info.line = 0;
    marmot_last_error_info.function = nullptr;
}

void marmot_set_error_ex(marmot_error_t error, const char *detail, const char *file, int line, const char *function) {
    marmot_last_error = error;
    marmot_last_error_info.code = error;
    marmot_last_error_info.file = file;
    marmot_last_error_info.line = line;
    marmot_last_error_info.function = function;

    if (detail != nullptr) {
        snprintf(marmot_last_error_detail, sizeof(marmot_last_error_detail), "%s", detail);
        snprintf(marmot_last_error_info.message, sizeof(marmot_last_error_info.message), "%s", detail);
    } else {
        marmot_last_error_detail[0] = '\0';
        marmot_last_error_info.message[0] = '\0';
    }
}

const marmot_error_info_t *marmot_get_last_error_info(void) {
    return &marmot_last_error_info;
}
