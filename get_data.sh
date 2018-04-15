curl yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz > train_images.gz
curl yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz > train_labels.gz

gzip -d train_images.gz 
gzip -d train_labels.gz 


curl yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz > test_images.gz 
curl yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz > test_labels.gz

gzip -d test_images.gz 
gzip -d test_labels.gz 

