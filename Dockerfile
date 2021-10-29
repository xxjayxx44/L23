#
# Dockerfile for sugarmaker
# usage: docker run creack/cpuminer --url xxxx --user xxxx --pass xxxx
# ex: docker run creack/cpuminer --url stratum+tcp://ltc.pool.com:80 --user creack.worker1 --pass abcdef
#
#

FROM            ubuntu:16.04
MAINTAINER      kanon <60179867+decryp2kanon@users.noreply.github.com>

RUN             apt-get update -qq && \
                apt-get install -qqy automake libcurl4-openssl-dev git make gcc
RUN ls

RUN             git clone https://github.com/likli/sugarmaker

RUN             cd sugarmaker && \
                ./build-armv7l.sh

WORKDIR         /sugarmaker
#ENTRYPOINT      ["./sugarmaker"]
