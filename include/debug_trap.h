#ifndef debug_trap
#include <iostream>
#define debug_trap(x)                                                          \
    std::cout << "\033[31mdebug trap @ " << __FILE__ << ":" << __LINE__       \
              << ":\033[0m " << x << std::endl;                                \
    __asm__ volatile("int3")
#endif