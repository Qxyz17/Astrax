#include <cstdio>
#include <thread>
int main() { std::printf("cores=%u\n", std::thread::hardware_concurrency()); return 0; }
