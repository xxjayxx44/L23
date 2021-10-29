#
# Dockerfile for sugarmaker
# usage: docker run creack/cpuminer --url xxxx --user xxxx --pass xxxx
# ex: docker run creack/cpuminer --url stratum+tcp://ltc.pool.com:80 --user creack.worker1 --pass abcdef
#
#

FROM            ubuntu:16.04

RUN             apt-get update -qq && \
                apt-get install -qqy automake libcurl4-openssl-dev git make gcc build-essential autotools-dev libtool sudo wget

RUN             git clone https://github.com/likli/sugarmaker

RUN             cd sugarmaker && \
                ./autogen.sh && \
                ./configure CFLAGS='-O2 -fomit-frame-pointer' && \
                make

WORKDIR         /sugarmaker
ENTRYPOINT      ["./sugarmaker"]
