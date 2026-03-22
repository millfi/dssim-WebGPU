## dssimのWebGPU dawn nativeによる高速化の試み
- アプリ名: `dssim-WebGPU`
- 入力: PNGのみ。`libpng`でデコードをしています。ベンチマークの際は、非圧縮PNGを用いてください。
- C++20で実装
- 現状: dsssimのロジックのWGSLへの移植は成功。高速化はされておらず、むしろ3倍遅い。また、dssim3.4のロジックを移植しているが、完全にdssim3.4と同じスコアを返すわけではなく、以下のようにわずかに異なる値になる。
`tests/1440p.png` vs `tests/1440p.jxl.png`:
- Reference (dssim v3.4.0): `0.00044658`
- WebGPU : `0.00044330`

## 備考
- 実行バイナリの`d3dcompiler_47`はWindowsにインストールされてることが多いので削除可能
- 何度も実行すると`[profiling]`の`Readback processing time`が速くなることがわかった。
