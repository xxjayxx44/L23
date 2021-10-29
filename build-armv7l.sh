# try on virtualbox ubuntu 16.04
# https://lxadm.com/Static_compilation_of_cpuminer

# CLEAN
make distclean || echo clean
rm -f config.status

# DEPENDS

## OPENSSL
# wget https://www.openssl.org/source/openssl-1.1.0g.tar.gz
# tar -xvzf openssl-1.1.0g.tar.gz
# cd openssl-1.1.0g/
# ./config no-shared
# make -j$(nproc)
# sudo make install
# cd ..

## CURL
# wget https://github.com/curl/curl/releases/download/curl-7_57_0/curl-7.57.0.tar.gz
# tar -xvzf curl-7.57.0.tar.gz
# cd curl-7.57.0/
# .buildconf | grep "buildconf: OK"
# ./configure --disable-shared | grep "Static=yes"
# make -j$(nproc)
# sudo make install
# cd ..

# BUILD
./autogen.sh
# CFLAGS="-Wall -O2 -fomit-frame-pointer" ./configure
# ./configure --with-curl="/usr/local/" --with-crypto="/usr/local/" CFLAGS="-Wall -O2 -fomit-frame-pointer" LDFLAGS="-static -I/usr/local/lib/ -L/usr/local/lib/libcrypto.a" LIBS="-lssl -lcrypto -lz -lpthread -ldl" CFLAGS="-DCURL_STATICLIB" --with-crypto
./configure --with-curl="/usr/local/" --with-crypto="/usr/local/" CFLAGS="-Wall -O2 -fomit-frame-pointer" LDFLAGS="-static" LIBS="-ldl -lz"
make
strip -s sugarmaker

# CHECK STATIC
file sugarmaker | grep "statically linked"

# PACKAGE
RELEASE=sugarmaker-v2.5.0-sugar4-armv7l
rm -rf $RELEASE
mkdir $RELEASE
cp ./mining-script/sh/*.sh $RELEASE/
cp sugarmaker $RELEASE/

# SIGN
zip -X $RELEASE/$RELEASE.zip $RELEASE/*
sha256sum $RELEASE/$RELEASE.zip > $RELEASE/$RELEASE
gpg --digest-algo sha256 --clearsign $RELEASE/$RELEASE
rm $RELEASE/$RELEASE && cat $RELEASE/$RELEASE.asc
