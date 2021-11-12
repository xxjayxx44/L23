# CLEAN
make distclean || echo clean
rm -f config.status

# BUILD
./autogen.sh
./configure --with-curl="/usr/local/" --with-crypto="/usr/local/" CFLAGS="-Wall -O2 -fomit-frame-pointer" CXXFLAGS="$CFLAGS -std=gnu++11" LDFLAGS="-static" LIBS="-ldl -lz"
make -j$(nproc)
strip -s sugarmaker

# CHECK STATIC
file sugarmaker | grep "dynamically linked"
