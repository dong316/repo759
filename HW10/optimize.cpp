#include "optimize.h"

data_t *get_vec_start(vec *v) {
    return v->data;
}

// optimize1 = baseline / reduce4
void optimize1(vec *v, data_t *dest) {
    size_t n = v->len;
    data_t *d = get_vec_start(v);
    data_t acc = IDENT;

    for (size_t i = 0; i < n; i++) {
        acc = acc OP d[i];
    }

    *dest = acc;
}

// optimize2 = unroll 2x1
void optimize2(vec *v, data_t *dest) {
    size_t n = v->len;
    data_t *d = get_vec_start(v);
    data_t acc = IDENT;
    size_t i = 0;

    for (; i + 1 < n; i += 2) {
        acc = (acc OP d[i]) OP d[i + 1];
    }

    for (; i < n; i++) {
        acc = acc OP d[i];
    }

    *dest = acc;
}

// optimize3 = unroll 2x1a
void optimize3(vec *v, data_t *dest) {
    size_t n = v->len;
    data_t *d = get_vec_start(v);
    data_t acc = IDENT;
    size_t i = 0;

    for (; i + 1 < n; i += 2) {
        acc = acc OP (d[i] OP d[i + 1]);
    }

    for (; i < n; i++) {
        acc = acc OP d[i];
    }

    *dest = acc;
}

// optimize4 = unroll 2x2
void optimize4(vec *v, data_t *dest) {
    size_t n = v->len;
    data_t *d = get_vec_start(v);
    data_t acc0 = IDENT;
    data_t acc1 = IDENT;
    size_t i = 0;

    for (; i + 1 < n; i += 2) {
        acc0 = acc0 OP d[i];
        acc1 = acc1 OP d[i + 1];
    }

    for (; i < n; i++) {
        acc0 = acc0 OP d[i];
    }

    *dest = acc0 OP acc1;
}

// optimize5 = K=3, L=3 style
void optimize5(vec *v, data_t *dest) {
    size_t n = v->len;
    data_t *d = get_vec_start(v);
    data_t acc0 = IDENT;
    data_t acc1 = IDENT;
    data_t acc2 = IDENT;
    size_t i = 0;

    for (; i + 2 < n; i += 3) {
        acc0 = acc0 OP d[i];
        acc1 = acc1 OP d[i + 1];
        acc2 = acc2 OP d[i + 2];
    }

    for (; i < n; i++) {
        acc0 = acc0 OP d[i];
    }

    *dest = (acc0 OP acc1) OP acc2;
}