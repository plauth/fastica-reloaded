default:
	clang++ main.cpp -std=c++11 -I/usr/local/include/ -L/usr/local/lib/ -lboost_program_options -lsndfile -o fastica-reloaded
