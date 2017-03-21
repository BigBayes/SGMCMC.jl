gcc -Wall -fPIC -march=native -std=c99 -O3 -c c_funcs.c
gcc -shared -o libcjulia.so -Wl,-soname,libcjulia.so.1 -o libcjulia.so.1.0 c_funcs.o
rm -rf libcjulia.so.1 libcjulia.so
ln -s libcjulia.so.1.0 libcjulia.so
ln -s libcjulia.so.1.0 libcjulia.so.1
