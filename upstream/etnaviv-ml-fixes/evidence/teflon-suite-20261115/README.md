# Mesa's test_teflon on the NPU, 2026-11-15, upstream main 8a9009e0482 + 9 patches (libteflon b6c2a28d)

One npu-attempt per family (research/port/teflon-suite-ladder.sh), all in boot e67a48e7, run
runs/teflon-suite-20261115-145342. No kernel messages. Left out: teflon/ds_cnn.0014 and ds_cnn.whole (int8
FULLY_CONNECTED 276 -> 12 hung the NPU, incidents/2026-11-15-0745-TFS-ds_cnn) and the 32 tests of Mesa's skip
list for this NPU (src/etnaviv/ci/etnaviv-vipnano-skips.txt). The earlier run with 8 patches is in 8-patches/
(1502 passed, 35 failed: micronet_vww.whole failed there and passes with patch 9).

Baseline = unpatched main (lib 385f828f), rerun of the unexpected failures (baseline/, baseline-compare-*.tsv):
every one of them fails without the series as well (the two movenet whole models abort there: the SUBTRACT
assertion patch 8 removes). The unpatched library hung the NPU in micronet_vww.whole
(incidents/2026-11-15-1438-TFB-micronet_vww).

| Family | Passed | Failed | Failed tests (* = in Mesa's expected-failure list) |
|---|---|---|---|
| deeplabv3 | 65 | 8 | 0004* 0009 0011* 0022* 0052 0056 0060 whole*  |
| ds_cnn | 15 | 0 |  |
| efficientdet_lite0 | 266 | 1 | whole  |
| inception_v1 | 83 | 1 | whole  |
| micronet_ad_large | 15 | 0 |  |
| micronet_vww | 75 | 0 |  |
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
| total | 1503 | 34 | |
