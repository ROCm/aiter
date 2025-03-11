#include "utils.h"
#include <cassert>

int main(){
    auto res = executeCmd("echo hello");
    // assert(res.first == "hello");
    assert(res.second == 0);

    auto lib = SharedLibrary("math_test.so");
    int c = 0;
    lib.call("call", 1, 1, &c);
    assert(c == 2);
    return 0;
}