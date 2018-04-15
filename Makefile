src = *.cpp 
obj = $(src:.cpp=.o)
CC = gcc
LD_LIBRARY_PATH=gsl_priv/lib #/usr/local/lib
export LD_LIBRARY_PATH
export PKG_CONFIG_PATH=gsl_priv/lib/pkgconfig/gsl.pc
LDFLAGS = -lstdc++ -lgsl -lgslcblas -lm -L gsl_priv/lib

build: $(src)
	$(CC) -c $(src) -I gsl_priv/include
	$(CC)  $(obj) $(LDFLAGS) -o a.out

.PHONY: clean
clean: 
	rm -f $(obj) myprog








