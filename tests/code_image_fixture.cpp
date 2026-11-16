#if defined(_WIN32)
#define MGC_TEST_EXPORT extern "C" __declspec(dllexport)
#else
#define MGC_TEST_EXPORT extern "C" __attribute__((visibility("default")))
#endif
MGC_TEST_EXPORT int mgc_code_image_fixture() { return 42; }
