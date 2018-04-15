LD_LIBRARY_PATH=gsl_priv/lib #/usr/local/lib
export LD_LIBRARY_PATH
export PKG_CONFIG_PATH=gsl_priv/lib/pkgconfig/gsl.pc

# make build 
CPPUTEST_HOME = ./cpputest

# -I points to header files for GSL library (non root privileged users) 
gcc -Wall -Wextra -I cpputest/include -I gsl_priv/include -c -g *.cpp 

# lstdc++ for C++ standard libraries 
# -lgsl, -lm & -lgslcblas for GSL library 
#  -L points to library for GSL (non root based libraries) 
gcc -static *.o -lstdc++ -lgsl -lgslcblas -lm -L gsl_priv/lib -L cpputest/lib -o a.out










