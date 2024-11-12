# Simple benchmark for LRU caches

## How to run
```
ya make -r
./simple_bench -s 1000000 -t 1 -d 100
```

## Options:
```
--cache-size <size_t> or -s - cache size. Default: -
--mean <double> or -m - mean of normal distribution. Default: 0.0
--stddev <double> or -d - standard deviation of normal distribution. Default: equal to cache size
--thread-count <size_t> or -t - thread count. Default: 1
```
