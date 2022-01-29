# CLEAN
rm -f config.status

# DEPENDS
cd deps-linux64/
./deps-linux64.sh
cd ..

# BUILD
./autogen.sh
./configure CFLAGS="-Wall -O2 -fomit-frame-pointer" LDFLAGS="-static" CXXFLAGS="$CFLAGS -std=gnu++11" --with-curl=/usr/local/ --with-crypto
make
