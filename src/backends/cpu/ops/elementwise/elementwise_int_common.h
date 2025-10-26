#ifndef CPU_ELEMENTWISE_INT_UTILS_H
#define CPU_ELEMENTWISE_INT_UTILS_H

#include <stdint.h>

static inline unsigned cpu_ew_shift_amount(int64_t raw, unsigned bits) {
    if (raw <= 0) {
        return 0;
    }
    if ((uint64_t)raw >= bits) {
        return bits;
    }
    return (unsigned)raw;
}

#endif // CPU_ELEMENTWISE_INT_UTILS_H
