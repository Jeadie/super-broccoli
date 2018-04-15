mkdir gsl_lib
cd gsl 
./configure --prefix=gsl_lib
make
make check
make install 
cd ../
