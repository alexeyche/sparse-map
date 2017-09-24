FROM sm-base

WORKDIR /sparse-map-build

ARG SM_PATH 
ENV SM_PATH ${SM_PATH:-/usr/share/sparse-map}

ARG SMB_PATH 
ENV SMB_PATH ${SMB_PATH:-/usr/share/sparse-map-build}


#RUN mkdir $SMB_PATH; \
#	cd $SMB_PATH; \
#    cmake $SM_PATH; \
#    make -j
#
#RUN ln -s /usr/share/sparse-map-build/test-sparse-map /

