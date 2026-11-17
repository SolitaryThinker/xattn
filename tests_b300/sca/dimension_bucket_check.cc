// Host-only value and multi-translation-unit conformance for bucket selection.
// Compile this file twice, once with XATTN_BUCKET_HELPER_TU, then link the objects.
#include "flash_sca/native/dimension_bucket.h"
#include "flash_sca/native/dimension_bucket.h"

#include <cstdint>

using BucketFunction = int (*)(int64_t);
using xattn::ops::flash_sca_native::QkBucket;
using xattn::ops::flash_sca_native::ValueBucket;

#ifdef XATTN_BUCKET_HELPER_TU

int QkBucketFromSecondTu(int64_t dim) { return QkBucket(dim); }
int ValueBucketFromSecondTu(int64_t dim) { return ValueBucket(dim); }
BucketFunction QkBucketAddressFromSecondTu() { return &QkBucket; }
BucketFunction ValueBucketAddressFromSecondTu() { return &ValueBucket; }

#else

#include <cstdio>
#include <limits>

int QkBucketFromSecondTu(int64_t dim);
int ValueBucketFromSecondTu(int64_t dim);
BucketFunction QkBucketAddressFromSecondTu();
BucketFunction ValueBucketAddressFromSecondTu();

static_assert(QkBucket(8) == 64 && QkBucket(64) == 64);
static_assert(QkBucket(65) == 128 && QkBucket(128) == 128);
static_assert(QkBucket(129) == 256 && QkBucket(256) == 256);
static_assert(ValueBucket(96) == 128 && ValueBucket(160) == 256);
// Bucket selection intentionally does not reject out-of-contract dimensions.
static_assert(QkBucket(0) == 64 && ValueBucket(257) == 256);

int main() {
  int cases = 0;
  int checks = 0;
  int failures = 0;
  const auto expect = [&](bool pass) {
    ++checks;
    if (!pass) ++failures;
  };
  const auto test_dimension = [&](int64_t dim) {
    ++cases;
    // The previous dispatch expression, retained independently as the ruler.
    const int expected = dim <= 64 ? 64 : dim <= 128 ? 128 : 256;
    expect(QkBucket(dim) == expected);
    expect(ValueBucket(dim) == expected);
    expect(QkBucketFromSecondTu(dim) == expected);
    expect(ValueBucketFromSecondTu(dim) == expected);
  };
  for (int64_t dim = -1024; dim <= 2048; ++dim) test_dimension(dim);
  const int64_t edges[] = {
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::min() + 1,
      -65536, -1025, 2049, 4096,
      std::numeric_limits<int64_t>::max() - 1,
      std::numeric_limits<int64_t>::max(),
  };
  for (int64_t dim : edges) test_dimension(dim);
  // Inline functions denote one entity across translation units, even at O0.
  expect(QkBucketAddressFromSecondTu() == &QkBucket);
  expect(ValueBucketAddressFromSecondTu() == &ValueBucket);
  std::printf("{\"cases\":%d,\"checks\":%d,\"failures\":%d,\"passed\":%s}\n",
              cases, checks, failures, failures == 0 ? "true" : "false");
  return failures == 0 ? 0 : 1;
}

#endif
