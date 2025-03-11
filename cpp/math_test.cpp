// #include "math_test.h"
extern "C" {
    void call(int a, int b, int* c);
}

void call(int a, int b, int* c){
    *c = a + b;
}

