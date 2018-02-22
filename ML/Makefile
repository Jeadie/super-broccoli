LD_LIBRARY_PATH=gsl_priv/lib #/usr/local/lib
export LD_LIBRARY_PATH
export PKG_CONFIG_PATH=gsl_priv/lib/pkgconfig/gsl.pc

# -I points to header files for GSL library (non root privileged users) 
gcc -Wall -Wextra -I gsl_priv/include -c -g Digit_Recog_ANN.cpp data_parser.cpp data_parser.cpp

echo "###############################"
# lstdc++ for C++ standard libraries 
# -lgsl, -lm & -lgslcblas for GSL library 
#  -L points to library for GSL (non root based libraries) 
gcc -static Digit_Recog_ANN.o data_parser.o -lstdc++ -lgsl -lgslcblas -lm -L gsl_priv/lib 
