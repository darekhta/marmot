#include "marmot/calibration.h"

#include <stdio.h>
#include <stdlib.h>

#include <ctype.h>
#include <string.h>
#include <sys/utsname.h>

#include "yyjson.h"

static void sanitize_component(char *s) {
    if (s == nullptr) {
        return;
    }
    for (char *p = s; *p != '\0'; ++p) {
        unsigned char c = (unsigned char)(*p);
        if (isalnum(c)) {
            *p = (char)tolower(c);
        } else {
            *p = '_';
        }
    }
}

bool marmot_calibration_make_key(
    const char *prefix, const char *device_name, uint32_t cores, char *buf, size_t buf_sz
) {
    if (buf == nullptr || buf_sz == 0 || prefix == nullptr || prefix[0] == '\0') {
        return false;
    }
    struct utsname uts = {0};
    const char *sys = "unknown";
    const char *rel = "0";
    if (uname(&uts) == 0) {
        sys = uts.sysname;
        rel = uts.release;
    }
    char name_buf[128] = {0};
    if (device_name != nullptr && device_name[0] != '\0') {
        strncpy(name_buf, device_name, sizeof(name_buf) - 1);
    } else {
        strncpy(name_buf, "device", sizeof(name_buf) - 1);
    }
    sanitize_component(name_buf);
    char sys_buf[64] = {0};
    strncpy(sys_buf, sys, sizeof(sys_buf) - 1);
    sanitize_component(sys_buf);
    char rel_buf[64] = {0};
    strncpy(rel_buf, rel, sizeof(rel_buf) - 1);
    sanitize_component(rel_buf);
    if (cores == 0) {
        cores = 1;
    }
    int written = snprintf(buf, buf_sz, "%s_%s_%ucores_%s_%s", prefix, name_buf, cores, sys_buf, rel_buf);
    return written > 0 && (size_t)written < buf_sz;
}

static bool calibration_path(char *buf, size_t buf_sz) {
    if (buf == nullptr || buf_sz == 0) {
        return false;
    }
    const char *home = getenv("HOME");
    if (home == nullptr || home[0] == '\0') {
        return false;
    }
    int written = snprintf(buf, buf_sz, "%s/.cache/marmot/calibration.json", home);
    return written > 0 && (size_t)written < buf_sz;
}

static bool read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    fclose(f);
    return true;
}

static bool parse_calibration_obj(yyjson_val *obj, marmot_calibration_t *calib) {
    if (obj == nullptr || calib == nullptr || !yyjson_is_obj(obj)) {
        return false;
    }
    bool found = false;
    yyjson_val *root = obj;
    yyjson_val *nested = yyjson_obj_get(obj, "calibration");
    if (nested != nullptr && yyjson_is_obj(nested)) {
        root = nested;
    }
    yyjson_val *v = yyjson_obj_get(root, "effective_tflops_fp32");
    if (v != nullptr && yyjson_is_num(v)) {
        calib->effective_tflops_fp32 = (float)yyjson_get_real(v);
        found = true;
    }
    v = yyjson_obj_get(root, "effective_tflops_fp16");
    if (v != nullptr && yyjson_is_num(v)) {
        calib->effective_tflops_fp16 = (float)yyjson_get_real(v);
        found = true;
    }
    v = yyjson_obj_get(root, "effective_gbps");
    if (v != nullptr && yyjson_is_num(v)) {
        calib->effective_gbps = (float)yyjson_get_real(v);
        found = true;
    }
    v = yyjson_obj_get(root, "launch_overhead_us");
    if (v != nullptr && yyjson_is_num(v)) {
        calib->launch_overhead_us = (float)yyjson_get_real(v);
        found = true;
    }
    v = yyjson_obj_get(root, "edge_penalty_alpha");
    if (v != nullptr && yyjson_is_num(v)) {
        calib->edge_penalty_alpha = (float)yyjson_get_real(v);
        found = true;
    }
    v = yyjson_obj_get(root, "dequant_us_q4k");
    if (v != nullptr && yyjson_is_num(v)) {
        calib->dequant_us_q4k = (float)yyjson_get_real(v);
        found = true;
    }
    v = yyjson_obj_get(root, "dequant_us_q5k");
    if (v != nullptr && yyjson_is_num(v)) {
        calib->dequant_us_q5k = (float)yyjson_get_real(v);
        found = true;
    }
    v = yyjson_obj_get(root, "dequant_us_q6k");
    if (v != nullptr && yyjson_is_num(v)) {
        calib->dequant_us_q6k = (float)yyjson_get_real(v);
        found = true;
    }
    v = yyjson_obj_get(root, "dequant_us_q8_0");
    if (v != nullptr && yyjson_is_num(v)) {
        calib->dequant_us_q8_0 = (float)yyjson_get_real(v);
        found = true;
    }
    v = yyjson_obj_get(root, "epilogue_scale");
    if (v != nullptr && yyjson_is_num(v)) {
        calib->epilogue_scale = (float)yyjson_get_real(v);
        found = true;
    }
    return found;
}

static bool match_device_key(yyjson_val *obj, const char *device_key) {
    if (obj == nullptr || device_key == nullptr) {
        return false;
    }
    yyjson_val *key = yyjson_obj_get(obj, "device_key");
    if (key != nullptr && yyjson_is_str(key)) {
        const char *val = yyjson_get_str(key);
        return val != nullptr && strcmp(val, device_key) == 0;
    }
    return false;
}

void marmot_calibration_defaults(marmot_calibration_t *calib) {
    if (calib == nullptr) {
        return;
    }
    *calib = (marmot_calibration_t){
        .effective_tflops_fp32 = 0.0f,
        .effective_tflops_fp16 = 0.0f,
        .effective_gbps = 0.0f,
        .launch_overhead_us = 0.0f,
        .edge_penalty_alpha = 0.0f,
        .dequant_us_q4k = 0.0f,
        .dequant_us_q5k = 0.0f,
        .dequant_us_q6k = 0.0f,
        .dequant_us_q8_0 = 0.0f,
        .epilogue_scale = 1.0f,
    };
}

bool marmot_calibration_load(const char *device_key, marmot_calibration_t *calib) {
    if (device_key == nullptr || device_key[0] == '\0' || calib == nullptr) {
        return false;
    }
    marmot_calibration_defaults(calib);
    char path[512];
    if (!calibration_path(path, sizeof(path))) {
        return false;
    }
    if (!read_file(path)) {
        return false;
    }

    yyjson_read_err err;
    yyjson_doc *doc = yyjson_read_file(path, 0, nullptr, &err);
    if (doc == nullptr) {
        return false;
    }
    yyjson_val *root = yyjson_doc_get_root(doc);
    bool found = false;
    if (yyjson_is_obj(root)) {
        yyjson_val *direct = yyjson_obj_get(root, device_key);
        if (direct != nullptr && parse_calibration_obj(direct, calib)) {
            found = true;
        }
        if (!found && match_device_key(root, device_key) && parse_calibration_obj(root, calib)) {
            found = true;
        }
        if (!found) {
            yyjson_val *cal = yyjson_obj_get(root, "calibration");
            if (cal != nullptr && match_device_key(root, device_key) && parse_calibration_obj(cal, calib)) {
                found = true;
            }
        }
        if (!found && yyjson_obj_size(root) > 0) {
            size_t idx = 0, max = 0;
            yyjson_val *key;
            yyjson_val *val;
            yyjson_obj_foreach(root, idx, max, key, val) {
                if (yyjson_is_str(key)) {
                    const char *k = yyjson_get_str(key);
                    if (k != nullptr && strcmp(k, device_key) == 0 && parse_calibration_obj(val, calib)) {
                        found = true;
                        break;
                    }
                }
            }
        }
    } else if (yyjson_is_arr(root)) {
        size_t idx = 0, max = 0;
        yyjson_val *val;
        yyjson_arr_foreach(root, idx, max, val) {
            if (match_device_key(val, device_key) && parse_calibration_obj(val, calib)) {
                found = true;
                break;
            }
        }
    }
    yyjson_doc_free(doc);
    return found;
}

void marmot_calibration_apply(const marmot_calibration_t *calib, marmot_device_caps_t *caps) {
    if (calib == nullptr || caps == nullptr) {
        return;
    }
    if (calib->effective_tflops_fp32 > 0.0f) {
        caps->peak_flops_tflops_fp32 = calib->effective_tflops_fp32;
    }
    if (calib->effective_tflops_fp16 > 0.0f) {
        caps->peak_flops_tflops_fp16 = calib->effective_tflops_fp16;
    }
    if (calib->effective_gbps > 0.0f) {
        caps->mem_bw_gbps = calib->effective_gbps;
    }
    if (calib->launch_overhead_us > 0.0f) {
        caps->launch_overhead_us = calib->launch_overhead_us;
    }
    if (calib->edge_penalty_alpha > 0.0f) {
        caps->edge_penalty_alpha = calib->edge_penalty_alpha;
    }
    if (calib->dequant_us_q4k > 0.0f) {
        caps->calib_dequant_us_q4k = calib->dequant_us_q4k;
    }
    if (calib->dequant_us_q5k > 0.0f) {
        caps->calib_dequant_us_q5k = calib->dequant_us_q5k;
    }
    if (calib->dequant_us_q6k > 0.0f) {
        caps->calib_dequant_us_q6k = calib->dequant_us_q6k;
    }
    if (calib->dequant_us_q8_0 > 0.0f) {
        caps->calib_dequant_us_q8_0 = calib->dequant_us_q8_0;
    }
    if (calib->epilogue_scale > 0.0f) {
        caps->calib_epilogue_scale = calib->epilogue_scale;
    }
}
