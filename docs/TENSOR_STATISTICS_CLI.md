# 独立张量统计示例（`mgc_tensor_statistics`）说明

`mgc_tensor_statistics` 示例在无参数运行时执行内置自检；传入参数后，可通过公共 Compute C ABI 对外置 JSON 夹具运行归一化与张量统计：

```sh
mgc_tensor_statistics --provider ncnn-cpu --fixture tests/data/statistics_v1.json --report report.json
```

## 1. 参数与后端选择

- `--provider` 支持 `cpu-reference` 与 `ncnn-cpu`。若当前构建未开启所选 provider，程序直接报错退出，不会静默降级切换到其他 domain。
- 无论外层调度 domain 选用 `cpu-reference` 还是 `ncnn-cpu`，本示例内部的统计与归一化运算均调用 CPU 参考算子执行。

## 2. 输入夹具校验规则

- `--fixture` 必须指向常规 UTF-8/ASCII JSON 文件（不接受符号链接），文件大小不超过 `2 MiB`。
- JSON 内容必须严格包含 `schema_version: 1`、正整数 `width` 与 `height`，以及长度恰好等于 `width * height` 的有限数值数组 `values`，元素总数最多 `65,536` 个。
- 数值在提交前转换为有限 IEEE 754 `float32`（允许范围内的正常舍入）；重复字段、未知字段、畸形格式及溢出数值一律拒绝。

## 3. 输出报告规则

- 仅当计算任务与独立数值校对全部成功后才创建 `--report` 文件；采用独占创建模式，绝不覆盖已存在的文件（写入中途若遇 I/O 失败可能保留局部文件并报错）。
- 报告记录输入夹具的 SHA-256、原生 `float32` 输入与输出的 SHA-256、字节序、实际选用的 domain、算子 provider、有限元素个数、最小值、最大值、总和及归一化结果。

## 适用范围与验证说明

本工具仅作为宿主端公共 C ABI 与算子功能校验示例，不构成 GPU 性能基准测试，也不作为真机硬件或正式发布资格依据。
