# Independent tensor statistics consumer

The `mgc_tensor_statistics` example keeps its existing built-in test when run
without arguments. Its file-driven form uses the public Compute C ABI:

```text
mgc_tensor_statistics --provider ncnn-cpu --fixture tests/data/statistics_v1.json --report report.json
```

`--provider` is either `cpu-reference` or `ncnn-cpu`; a provider that the build
does not support fails instead of silently choosing another domain. The
statistics and normalization operations in this example are CPU reference
kernels even when the runtime domain is `ncnn-cpu`. This is not a GPU or
performance benchmark.

The input is a regular, non-symbolic-link UTF-8/ASCII JSON file of at most
2 MiB, with exactly `schema_version: 1`, positive integer `width`, `height`, and
`values` containing exactly `width * height` finite numeric values. At most
65,536 float32 values are accepted. Values are converted to finite IEEE 754
float32 before submission; out-of-range values and malformed/duplicate fields
are rejected. Values within the float32 range may round during conversion.

The output is created exclusively after successful jobs and independent numeric
checks. An existing path is never replaced. A write failure can leave a partial
new file and is reported as an error. The report includes the fixture SHA-256,
native float32 input/output SHA-256, byte order, actual domain selection,
kernel provider, finite count, min, max, sum, and normalized values. These are
host-consumer results; no device or release qualification is implied.
