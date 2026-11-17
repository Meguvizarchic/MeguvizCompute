# Guarded worker host test

Configure this directory separately with `PACK_SOURCE` set to a complete
candidate checkout. It builds a separate CPU-reference core with the private
`MGC_TEST_GUARDED_HOST` injection flag. The default production core never
defines this flag. The harness runs the real `mgc_submit`, worker, ticket,
completion and drain paths with synthetic producer state; it does not prove
Android sync-file, AHardwareBuffer, Vulkan or driver behavior.

The `latency` mode records host synthetic gate-open-to-execute samples for
both producer-first and cleanup-first orderings. These measurements include
OS scheduling and are not device latency results.
