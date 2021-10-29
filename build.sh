# CLEAN
make distclean || echo clean
rm -f config.status

# BUILD
./autogen.sh
./configure CFLAGS="-Wall -O2 -fomit-frame-pointer" target_os="x86_64-pc-linux-gnu"
make -j$(nproc)
strip -s sugarmaker

# CHECK STATIC
file sugarmaker | grep "dynamically linked"
