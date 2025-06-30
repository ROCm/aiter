pa_ragged_test:
	hipcc pa_ragged_lib.so pa_ragged_test.cpp -std=c++17 -O0 -g -lgtest -lgtest_main -L/usr/local/lib -lfmt -lcrypto -o pa_ragged_test.out

pa_ragged_lib.so:
	hipcc pa_ragged.cpp -o pa_ragged_lib.so -fPIC -shared -I../include -std=c++17 -O3 -lfmt -lcrypto -ldl

pa_test:
	hipcc pa_lib.so pa_test.cpp -std=c++17 -O0 -g -lgtest -lgtest_main -L/usr/local/lib -lfmt -lcrypto -o pa_test.out

pa_lib.so:
	hipcc pa.cpp -o pa_lib.so -fPIC -shared -I../include -std=c++17 -O3 -lfmt -lcrypto -ldl

clean:
	rm -f pa_test.out pa_ragged_test.out pa_ragged_lib.so pa_lib.so