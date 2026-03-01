## dssimのWebGPU dawn nativeによる高速化の試み
- アプリ名: `dssim_gpu_dawn_checksum`
- 入力: PNGのみ。`libpng`でデコードをしています。ベンチマークの際は、非圧縮PNGを用いてください。
- C++20で実装
- 現状: dsssimのロジックのWGSLへの移植は成功。高速化はされておらず、むしろ3倍遅い。また、dssim3.4のロジックを移植しているが、完全にdssim3.4と同じスコアを返すわけではなく、以下のようにわずかに異なる値になる。
- Reference (`dssim` CLI): `0.00044658`
- WebGPU (`dssim_gpu_dawn_checksum`): `0.00044330`
## CMakeの自動依存パッケージインストールの問題点
どういうわけか、
```powershell
cmake -S . -B build `
  -DDSSIM_DAWN_ROOT="<path-to-dawn-src>" `
  -DDSSIM_DAWN_OUT_DIR="$(Resolve-Path .\third_party\dawn\out\Release)"

cmake --build build --config Release --target dssim_gpu_dawn_checksum
```
はうまくいきません。これはcodex-5.2-mediumの作業により生成されましたが、dawnの依存関係がよくわからず、`git clone→CMake自動依存インストール→build`が失敗します。
## 実行コマンドが冗長すぎる
```
$env:PATH = "$(Resolve-Path .\third_party\dawn\out\Release);$env:PATH"

.\build\src_gpu\Release\dssim_gpu_dawn_checksum.exe `
  .\tests\gray-profile.png .\tests\gray-profile2.png `
  --out .\out\gpu.json `
  --debug-dump-dir .\out\debug
```
`$env:PATH = "$(Resolve-Path .\third_party\dawn\out\Release);$env:PATH"`は明らかに不要です。原因はdllのパスの問題です。
