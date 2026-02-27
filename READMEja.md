## dssimのWebGPU dawn nativeによる高速化の試み
buildと実行コマンドは`README.md`を参照してください。
- アプリ名: `dssim_gpu_dawn_checksum`
- 入力: PNGのみ。`libpng`でデコードをしています。ベンチマークの際は、非圧縮PNGを用いてください。
- C++20で実装
- 現状: dsssimのロジックのWGSLへの移植は成功。高速化はされておらず、むしろ3倍遅い。また、dssim3.4のロジックを移植しているが、完全にdssim3.4と同じスコアを返すわけではなく、以下のようにわずかに異なる値になる。
- Reference (`dssim` CLI): `0.00044658`
- WebGPU (`dssim_gpu_dawn_checksum`): `0.00044330`