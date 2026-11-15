# Mesa's test_teflon on the NPU, 2026-11-15, upstream main 8a9009e0482 + 8 patches (libteflon a05298e9)

One npu-attempt per family (research/port/teflon-suite-ladder.sh). mobilenet_v1 and mobilenet_v2 ran in boot 64cd6b68,
the rest in boot 94e82477. No kernel messages in any of these runs. Left out: teflon/ds_cnn.0014 and ds_cnn.whole
(int8 FULLY_CONNECTED 276 -> 12 hung the NPU at 07:44, incidents/2026-11-15-0745-TFS-ds_cnn) and the 32 tests of
Mesa's skip list for this NPU (src/etnaviv/ci/etnaviv-vipnano-skips.txt).

| Family | Passed | Failed | Failed tests (* = in Mesa's expected-failure list) |
|---|---|---|---|
| deeplabv3 | 65 | 8 | 0004* 0009 0011* 0022* 0052 0056 0060 whole*  |
| ds_cnn | 15 | 0 |  |
| efficientdet_lite0 | 266 | 1 | whole  |
| inception_v1 | 83 | 1 | whole  |
| micronet_ad_large | 15 | 0 |  |
| micronet_vww | 74 | 1 | whole  |
| mobilenet_v1 | 32 | 0 |  |
| mobilenet_v2 | 66 | 0 |  |
| movenet_lightning | 151 | 4 | 0115 0118 0151 whole  |
| movenet_thunder | 151 | 4 | 0115 0118 0151 whole  |
| posenet_mobilenet_v1 | 29 | 4 | 0003* 0007* 0011* whole*  |
| ssdlite_mobiledet | 109 | 0 |  |
| ssd_mobilenet_v2 | 108 | 3 | 0011 0052 whole  |
| ssd_mobilenet_v3 | 174 | 1 | whole  |
| wav2letter | 24 | 0 |  |
| yolox_nano | 140 | 8 | 0001 0104 0114 0124 0127 0130 0133 whole  |
| total | 1502 | 35 | |
