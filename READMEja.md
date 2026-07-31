# dssim-WebGPU

DSSIM画像・動画比較アルゴリズムをC++20とVulkanで実装した高速化版です。
コマンドラインとビルドの互換性を保つため、実行ファイル名とCMakeターゲット名は
従来のまま維持しています。

英語版は [README.md](README.md) を参照してください。

## 必要環境

- 次の機能に対応するVulkan GPUとドライバーを搭載したWindows
  - Vulkan 1.3
  - `VK_EXT_shader_object`
  - `VK_KHR_push_descriptor`
  - Vulkan 1.3の`synchronization2`および`dynamicRendering`機能
- Vulkan loaderライブラリとヘッダー、および`glslc`を含むVulkan SDK
- PowerShell
- CMake 3.24以降
- C++20対応コンパイラ
- 動画比較では、Vulkan Video decode queueと、各入力codecに対応する
  `VK_KHR_video_decode_h264`、`VK_KHR_video_decode_h265`、
  `VK_KHR_video_decode_vp9`、または`VK_KHR_video_decode_av1`

トップレベルのCMake設定が自動的にC++20を選択するため、ビルド時に
C++標準の追加フラグを指定する必要はありません。

CMakeは`find_package(Vulkan REQUIRED COMPONENTS glslc)`でSDKを検出します。
通常のVulkan SDKインストールで設定される`VULKAN_SDK`をCMakeが利用できます。
SDKを自動取得する処理はありません。

入力は、同じ幅・高さのPNG画像、または2本の動画です。動画コンテナは拡張子
`.mp4`、`.m4v`、`.mov`、`.mkv`、`.webm`で判定します。FFmpegのVulkan Video
デコーダー（H.264/HEVC/VP9/AV1）から`AV_PIX_FMT_VULKAN`フレームを受け取り、
NV12またはP010のVulkan imageをGPU内でRGBA8へ変換します。デコードフレームの
CPU readbackはありません。

## ビルド

リポジトリ内のコマンドはすべてPowerShellから実行します。PNGだけを比較する場合も、
ターゲットのビルドには最小構成FFmpeg Vulkan Videoの開発ファイルが必要です。
Windows向けheader、import library、DLLは`third_party/ffmpeg-8.1.2-shared`に
同梱されるため、clone直後でもnetwork接続なしでbuildできます。GPU targetは次の
scriptでbuildします。

```powershell
& .\tools\build_gpu.ps1
```

同等の手動コマンドは次のとおりです。

```powershell
& cmake -S . -B build
& cmake --build build --config Release --target dssim_webgpu
```

同梱FFmpeg directoryがないか不完全な場合、`build_gpu.ps1`は
`build_ffmpeg_minimal.ps1`で再生成します。この復旧経路では完全版vcpkgがあれば
それを使用し、なければrepository内へvcpkgをdownloadするためnetwork接続が必要です。

実行ファイルは次の場所に生成されます。

```text
build\src_gpu\Release\dssim-WebGPU.exe
```

ルートのCMake設定は従来のconfigureコマンドとの互換性を維持し、
GPUターゲットの定義を`src_gpu/CMakeLists.txt`へ委譲します。ビルド時に
`glslc`が`src_gpu/shaders`のGLSLコンピュートシェーダーをSPIR-Vへ
コンパイルし、実行ファイルの隣へ配置します。

```text
build\src_gpu\Release\shaders\*.spv
```

## 1ペアを比較する

```powershell
& .\build\src_gpu\Release\dssim-WebGPU.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png
```

標準出力にスコアと比較対象のパスが表示されます。

```text
0.00328441    .\tests\laptop.q24.jpegli.jpg.png
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
`--debug-dump-dir <ディレクトリ>` を指定します。このオプションはPNG比較用です。

## 動画を比較する

画像ペアと同様に、2本の動画パスを指定します。

```powershell
& .\build\src_gpu\Release\dssim-WebGPU.exe `
    .\benchmark\x264_medium_g40_fastdecode_crf40.mp4 `
    .\benchmark\3s.webm `
    --profiling `
    --csv .\out\video_scores.csv
```

両方の入力が動画である必要があります。対応するデコードフレームの幅と高さ、
および両動画のデコードフレーム数が一致しなければエラーになります。比較は
0始まりのデコード順で行い、timestampによる位置合わせ、retime、resampleは
行いません。

処理中はstderrへFPS、現在in-flightのpipeline depth、設定されたpipeline
capacity、処理フレーム数、経過時間、直前フレームのDSSIM、累積平均DSSIMを
表示します。完了時はstdoutへ全フレームペアの平均DSSIMと`frames=<N>`を
表示します。

```text
0.06837245    .\benchmark\3s.webm    frames=180
```

`--csv <path>`は`time_seconds,frame_number,dssim`を、1フレームペアにつき1行
出力します。`time_seconds`は1本目の動画の先頭デコードフレームを0秒とした
timestampです。`--out <json>`は集約結果と、各フレームの値を合計した詳細な
profiling値を出力します。

2本の動画はそれぞれ専用のFFmpeg decode threadで処理され、比較pipelineへ
渡されます。同時処理するフレームペア数は、正の整数を取る
`--pipeline-depth <N>`で指定でき、既定値は3です。起動時に各codecと選択した
Vulkan Video queue familyを表示します。可能なら2本を別々の対応queue familyへ
割り当てますが、両方がAV1の場合はprimary queue familyを共有します。

## 固定の複数ペアベンチマーク

実行可能なベンチマークリストとして `tests/test_pairs.txt` を使用します。
空行以外の各行には、タブ区切りで2つの画像パスを記述します。

ベンチマークは次の固定コマンドで実行します。

```powershell
Get-Content .\tests\test_pairs.txt |
    & .\build\src_gpu\Release\dssim-WebGPU.exe --stdin-pairs --profiling
```

`--stdin-pairs` ではVulkan instanceとdeviceを作成し、SPIR-Vを読み込んで
shader objectをプロセス内で一度だけ作成し、すべてのペアで再利用します。
画像解像度はpush constantsとディスパッチ数で指定されるため、解像度が
異なってもshader objectを作り直す必要はありません。

`--stdin-pairs` は `--out`、`--csv`、`--pipeline-depth`、
`--debug-dump-dir` と同時には使えません。

## 自動テスト

テストスクリプト`check_regression.ps1`は、Vulkan版のスコアをPATH上から解決したオリジナルの
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
- Vulkan版を1つの `--stdin-pairs` セッションで実行する
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
独立したVulkan timestamp query結果がミリ秒単位で表示されます。timestamp
query結果は、選択したcompute queueが対応している場合にのみ取得します。

セッション初期化:

- `session_init_pipeline_setup_ms`: pipeline layoutとshader objectの作成
- `session_init_resource_prep_ms`: セッション単位のリソース準備
- `session_init_gpu_submit_wait_ms`: セッション単位のGPU送信・待機
- `session_init_gpu_timestamp_ms`: セッション単位のVulkan timestamp query時間
- `session_init_cpu_postprocess_ms`: セッション単位のCPU後処理
- `session_init_other_ms`: 上記以外のセッション初期化

各比較:

- `pipeline_setup_ms`: 比較単位のシェーダー準備
- `resource_prep_ms`: バッファ作成、アップロード、resource binding準備
- `gpu_submit_wait_ms`: コマンド記録、サブミット、readback待機のCPU wall時間
- `gpu_timestamp_ms`: Vulkan timestamp queryで測定した実GPU実行時間
  （対応queueでのみ取得）
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
- `post_process_base_scale_ms`
- `post_process_remaining_scales_ms`
- `post_process_ms`

`dispatch_and_submit_ms` はCPU側のコマンド構築・送信時間であり、
純粋なシェーダー実行時間ではありません。`readback_ms` にはGPU完了待ちと
host readback時間が含まれます。CPUとGPUは非同期に重なるため、
`gpu_timestamp_ms` はwall-clock時間区分の合計には含まれません。
2つのscale別post-process項目も、並列集計時には互いに重なる独立時間です。
JSONのfield名は互換性のため維持しています。shader objectを使うため、
`create_pso_ms`と`create_bind_group_ms`は0になる想定です。shader objectと
pipeline layoutの処理は、対応する既存bucketへ計上します。timestamp queryは
任意機能であり、queueが非対応でも比較処理とwall-clock profilingは動作します。

## 現在の高速化設計

- shader objectはプロセス内で一度だけ作成し、すべての解像度で再利用する
- stageバッファは、それまでに処理した最大画像まで拡張し、各スケールと
  後続ペアで再利用する
- push descriptorによりdescriptor poolとdescriptor setの割り当てを省く
- デバッグ統計用リソースは通常のベンチマーク経路とは別にキャッシュする
- sRGBからlinearへの変換には256要素のルックアップテーブルを使用する
- 通常経路では2枚の入力をGPUで変換し、CPUへの途中round-tripなしで全scaleの
  画像ピラミッドを構築する
- デバッグ経路では中間scaleデータを保持・出力できるよう、CPUで並列に画素変換と
  画像ピラミッド生成を行う
- 通常実行ではSSIMマップだけをreadbackする

## 最適化とスコアの方針

現在の優先事項は、スコアを維持しながらend-to-end latencyを短縮することです。

- 同一画像は必ず `0.00000000` を出力する
- その他のペアは `dssim.exe` に対する相対誤差1%未満を維持する
- 浮動小数点精度や代数変形は、回帰チェックの許容範囲内でのみ変更できる
- blur weightおよびSSIM定数は変更しない

## Vulkan SDKとシェーダー

configureの前にVulkan SDKをインストールしてください。ビルドにはSDKの
loaderライブラリ、ヘッダー、`glslc`が必要で、いずれかが見つからない場合は
configureが明確なエラーで終了します。PowerShellからシェーダーコンパイラを
確認できます。

```powershell
& glslc --version
```

ビルド時に次のGLSLコンピュートシェーダーをSPIR-Vへコンパイルします。
アプリケーション起動時のシェーダーコンパイルは行いません。

- `rgba8_to_linear.comp`
- `vulkan_yuv_to_rgba8.comp`
- `downsample_2x2.comp`
- `lab_preprocess.comp`
- `stage0_absdiff.comp`
- `stage0_score.comp`

アルゴリズムの詳細を調べるための参照ソースは `src_reference/` に残しています。
自動回帰では、意図的にPATH上から選択された `dssim.exe` を使用します。
