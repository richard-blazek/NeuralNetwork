main: main.c layer.c network.c
	gcc -o main main.c layer.c network.c -lm -O2
