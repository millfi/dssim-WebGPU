# dssim-WebGPU

DSSIM画像比較アルゴリズムをC++20とWebGPUで実装した高速化版です。
ネイティブGPU実行ファイルはDawnとWGSLコンピュートシェーダーを使用します。

英語版は [README.md](README.md) を参照してください。

## 必要環境

- D3D12対応GPUを搭載したWindows
- PowerShell
- CMake 3.24以降
- C++20対応コンパイラ

トップレベルのCMake設定が自動的にC++20を選択するため、ビルド時に
C++標準の追加フラグを指定する必要はありません。

入力画像は両方ともPNG形式で、幅と高さが一致している必要があります。

## ビルド

リポジトリ内のコマンドはすべてPowerShellから実行します。

```powershell
& cmake -S . -B build
& cmake --build build --config Release --target dssim_webgpu
```

実行ファイルは次の場所に生成されます。

```text
build\src_gpu\Release\dssim-WebGPU.exe
```

## 1ペアを比較する

```powershell
& .\build\src_gpu\Release\dssim-WebGPU.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png
```

標準出力にスコアと比較対象のパスが表示されます。

```text
0.00328379    .\tests\laptop.q24.jpegli.jpg.png
```

タイミングを表示するには `--profiling` を追加します。

```powershell
& .\build\src_gpu\Release\dssim-WebGPU.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png `
    --profiling
```

スケールごとの結果と詳細なタイミングをJSONへ出力するには
`--out <json>` を使用します。

```powershell
& .\build\src_gpu\Release\dssim-WebGPU.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png `
    --profiling `
    --out .\out\gpu.json
```

スコア調査用の中間バッファを出力する場合は
`--debug-dump-dir <ディレクトリ>` を指定します。

## 固定の複数ペアベンチマーク

実行可能なベンチマークリストとして `tests/test_pairs.txt` を使用します。
空行以外の各行には、タブ区切りで2つの画像パスを記述します。

ベンチマークは次の固定コマンドで実行します。

```powershell
Get-Content .\tests\test_pairs.txt |
    & .\build\src_gpu\Release\dssim-WebGPU.exe --stdin-pairs --profiling
```

`--stdin-pairs` ではWebGPUデバイス、シェーダーモジュール、レイアウト、
PSOをプロセス内で一度だけ作成し、すべてのペアで再利用します。画像解像度は
uniformバッファとディスパッチ数で指定されるため、解像度が異なってもPSOを
作り直す必要はありません。

`--stdin-pairs` は `--out` および `--debug-dump-dir` と同時には使えません。

## スコア回帰の自動確認

回帰チェッカーは、WebGPU版のスコアをPATH上から解決したオリジナルの
`dssim.exe` と比較します。

```powershell
& .\tools\check_regression.ps1
```

実際に選択された参照実行ファイルは次のコマンドで確認できます。

```powershell
(Get-Command dssim.exe -CommandType Application).Source
```

回帰チェッカーは次の処理を行います。

- `tests/test_pairs.txt` の全ペアを読み込む
- WebGPU版を1つの `--stdin-pairs` セッションで実行する
- 各ペアについてオリジナルの `dssim.exe` を実行する
- 同一画像比較では `0.00000000` を要求する
- その他の比較では相対誤差1%未満を要求する
- 結果を表形式で表示し、違反があれば非ゼロで終了する

引数を上書きする場合は次のように実行します。

```powershell
& .\tools\check_regression.ps1 `
    -PairList .\tests\test_pairs.txt `
    -GpuExecutable .\build\src_gpu\Release\dssim-WebGPU.exe `
    -RelativeTolerance 0.01
```

スコアまたは性能に影響する変更の後と、最適化をコミットする前には必ず
このチェックを実行してください。

## プロファイリング出力

`--profiling` を指定すると、互いに重複しないwall-clock時間区分と、
独立したWebGPU Timestamp Query結果がミリ秒単位で表示されます。

セッション初期化:

- `session_init_pipeline_setup_ms`: シェーダー、Pipeline Layout、PSOの作成
- `session_init_resource_prep_ms`: セッション単位のリソース準備
- `session_init_gpu_submit_wait_ms`: セッション単位のGPU送信・待機
- `session_init_gpu_timestamp_ms`: セッション単位のGPU Timestamp Query時間
- `session_init_cpu_postprocess_ms`: セッション単位のCPU後処理
- `session_init_other_ms`: 上記以外のセッション初期化

各比較:

- `pipeline_setup_ms`: 比較単位のシェーダーとパイプライン準備
- `resource_prep_ms`: バッファ作成、アップロード、Bind Group作成
- `gpu_submit_wait_ms`: ディスパッチ、サブミット、readback/map待機のCPU wall時間
- `gpu_timestamp_ms`: WebGPU Timestamp Queryで測定した実GPU実行時間
- `cpu_postprocess_ms`: CPU側のスコア集計
- `other_ms`: 色変換、画像ピラミッド生成など、デコード完了後の未分類処理

`--out <json>` を使うと、`profiling` オブジェクトへ次の詳細項目が
出力されます。

- `decode_done_to_score_ms`
- `create_shader_module_ms`
- `create_pso_ms`
- `create_buffer_ms`
- `write_input_buffer_ms`
- `create_pipeline_layout_ms`
- `create_bind_group_ms`
- `dispatch_and_submit_ms`
- `readback_ms`
- `gpu_submit_wait_ms`
- `gpu_timestamp_ms`
- `post_process_ms`

`dispatch_and_submit_ms` はCPU側のコマンド構築・送信時間であり、
純粋なシェーダー実行時間ではありません。`readback_ms` にはGPU完了待ちと
マッピング待ちが含まれます。CPUとGPUは非同期に重なるため、
`gpu_timestamp_ms` はwall-clock時間区分の合計には含まれません。
プロファイリングにはWebGPU `TimestampQuery` feature対応adapterが必要です。

## 現在の高速化設計

- PSOはプロセス内で一度だけ作成し、すべての解像度で再利用する
- stageバッファとBind Groupは、それまでに処理した最大画像まで拡張し、
  各スケールと後続ペアで再利用する
- デバッグ統計用リソースは通常のベンチマーク経路とは別にキャッシュする
- sRGBからlinearへの変換には256要素のルックアップテーブルを使用する
- 2枚の入力画像を並列に色変換・縮小する
- CPU画素変換と画像ピラミッド生成は、すべての画像サイズで同じ並列経路を
  使用し、小画像専用のフォールバックを持たない
- 通常実行ではSSIMマップだけをreadbackする

## 最適化とスコアの方針

現在の優先事項は、スコアを維持しながらend-to-end latencyを短縮することです。

- 同一画像は必ず `0.00000000` を出力する
- その他のペアは `dssim.exe` に対する相対誤差1%未満を維持する
- 浮動小数点精度や代数変形は、回帰チェックの許容範囲内でのみ変更できる
- blur weightおよびSSIM定数は変更しない
- シェーダーは `@workgroup_size(16, 16, 1)` の2次元ディスパッチを維持し、
  ディスパッチ数を `(ceil(width / 16), ceil(height / 16), 1)` とする

## Dawnのセットアップ

Dawnが存在しない場合、CMakeは `tools/install_dawn.ps1` を自動実行します。
このスクリプトは `third_party/depot_tools` を準備し、Dawnを
`third_party/dawn` へ取得して、必要なDawnライブラリをビルドします。

自動インストールを無効にする場合:

```powershell
& cmake -S . -B build -DDSSIM_AUTO_INSTALL_DAWN=OFF
```

WebGPUターゲット自体を無効にする場合:

```powershell
& cmake -S . -B build -DDSSIM_ENABLE_DAWN_SAMPLE=OFF
```

アルゴリズムの詳細を調べるための参照ソースは `src_reference/` に残しています。
自動回帰では、意図的にPATH上から選択された `dssim.exe` を使用します。
