# dssim-Vulkan

DSSIM画像類似度評価アルゴリズムをC++20と[NoGraphicsAPI](https://github.com/sebbbi/NoGraphicsAPI)（Vulkan）で実装した高速化、動画対応を追加したものです。

## 必要環境

- 次の機能に対応するVulkan GPUとドライバーを搭載したWindows
  - Vulkan 1.4
  - `VK_EXT_descriptor_heap`
  - `VK_KHR_device_address_commands`, `VK_KHR_shader_untyped_pointers`, `VK_EXT_mesh_shader`
  - Vulkan 1.4の`synchronization2`および`dynamicRendering`機能
- Vulkan loaderライブラリとヘッダー、および`slangc` (2026.14.1+) / `spirv-val` (2026.3+)を含むVulkan SDK
- PowerShell
- CMake 3.24以降
- C++20対応コンパイラ
- 動画比較では、Vulkan Video decode queueと、各入力codecに対応する
  `VK_KHR_video_decode_h264`、`VK_KHR_video_decode_h265`、
  `VK_KHR_video_decode_vp9`、または`VK_KHR_video_decode_av1`

トップレベルのCMake設定が自動的にC++20を選択するため、ビルド時に
C++標準の追加フラグを指定する必要はありません。

CMakeは`find_package(Vulkan 1.4.357 REQUIRED)`でSDKを検出します。
通常のVulkan SDKインストールで設定される`VULKAN_SDK`をCMakeが利用できます。
SDKを自動取得する処理はありません。

入力は、同じ幅・高さのPNG画像、または2本の動画です。動画コンテナは拡張子
`.mp4`、`.m4v`、`.mov`、`.mkv`、`.webm`で判定します。FFmpegのVulkan Video
デコーダー（H.264/HEVC/VP9/AV1）から`AV_PIX_FMT_VULKAN`フレームを受け取り、
NV12またはP010のVulkan imageをGPU内でRGBA8へ変換します。デコードフレームの
CPU readbackはありません。

NoGraphicsAPIはthird_party/NoGraphicsAPIに固定コミットで同梱しています。
FFmpeg連携用の拡張は同ディレクトリのREADME.dssim.mdに記載しています。
coherentかつhost-visibleなdevice-localメモリ（Resizable BARまたはUMA）が必要です。
.compファイルはSlangで、-lang slangを指定してコンパイルします。
src_gpu/shaders/compute_root.hでC++とシェーダーの引数構造を共有します。
Vulkan SDK 1.4.357以降が必要です。

Windows用の固定版Slangをリポジトリ内に導入するには、次を実行します。

```powershell
& .\tools\setup_slang.ps1
```

C++ツールチェーンを読み込んだDeveloper PowerShellからビルドしてください。

## build
```powershell
& .\tools\build_gpu.ps1
```

or

```powershell
& cmake -S . -B build
& cmake --build build --config Release --target dssim_vulkan
```

`build_gpu.ps1`は展開前に固定SHA-256でarchiveを検証します。archiveがないか変更されて
いる場合はGitHubの`origin`にある現在のcommitから同じarchiveをdownloadし、検証後に
置換します。archive内容の再生成には`build_ffmpeg_minimal.ps1`を使用できます。
repository内へdownloadしたvcpkgのdownload fileとbinary cacheはどちらも
`third_party/vcpkg`配下へ保存され、user共通のAppData cacheは使用しません。

実行ファイルは次の場所に生成されます。

```text
build\src_gpu\Release\dssim-Vulkan.exe
```

ルートのCMake設定は従来のconfigureコマンドとの互換性を維持し、
GPUターゲットの定義を`src_gpu/CMakeLists.txt`へ委譲します。ビルド時に
`slangc` (2026.14.1+) / `spirv-val` (2026.3+)が`src_gpu/shaders`のSlangコンピュートシェーダーをSPIR-Vへ
コンパイルし、実行ファイルの隣へ配置します。

```text
build\src_gpu\Release\shaders\*.spv
```

## 1ペアを比較する

```powershell
& .\build\src_gpu\Release\dssim-Vulkan.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png
```

標準出力にスコアと比較対象のパスが表示されます。

```text
0.00328441    .\tests\laptop.q24.jpegli.jpg.png
```

タイミングを表示するには `--profiling` を追加します。

```powershell
& .\build\src_gpu\Release\dssim-Vulkan.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png `
    --profiling
```

スケールごとの結果と詳細なタイミングをJSONへ出力するには
`--out <json>` を使用します。

```powershell
& .\build\src_gpu\Release\dssim-Vulkan.exe `
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
& .\build\src_gpu\Release\dssim-Vulkan.exe `
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
    & .\build\src_gpu\Release\dssim-Vulkan.exe --stdin-pairs --profiling
```

`--stdin-pairs` ではVulkan instanceとdeviceを作成し、SPIR-Vを読み込んで
compute PSOをプロセス内で一度だけ作成し、すべてのペアで再利用します。
画像解像度はroot dataとディスパッチ数で指定されるため、解像度が
異なってもcompute PSOを作り直す必要はありません。

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

テストスクリプトは次の処理を行います。

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
    -GpuExecutable .\build\src_gpu\Release\dssim-Vulkan.exe `
    -RelativeTolerance 0.01
```


## プロファイリング出力

`--profiling` を指定すると、互いに重複しないwall-clock時間区分と、
独立したVulkan timestamp query結果がミリ秒単位で表示されます。timestamp
query結果は、選択したcompute queueが対応している場合にのみ取得します。

セッション初期化:

- `session_init_pipeline_setup_ms`: NoGraphicsAPI compute PSOの作成
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

JSONのフィールド名は互換性のため維持しています。NoGraphicsAPIのPSO作成時間は
従来のcreate_shader_module_msへ計上します。create_pso_ms、
create_pipeline_layout_ms、create_bind_group_msは0です。
gpu_timestamp_msは独立したGPU時間で、CPUのwall-clock合計には含めません。
timestamp query非対応のqueueでも比較とwall-clock profilingは動作します。

## 現在の高速化設計

- compute PSOはプロセス内で一度だけ作成し、すべての解像度で再利用する
- stageバッファは、それまでに処理した最大画像まで拡張し、各スケールと
  後続ペアで再利用する
- バッファはGPUアドレスをroot dataで渡し、動画の画像はdescriptor heapで参照する
- デバッグ統計用リソースは通常のベンチマーク経路とは別にキャッシュする
- sRGBからlinearへの変換には256要素のルックアップテーブルを使用する
- 通常経路では2枚の入力をGPUで変換し、CPUへの途中round-tripなしで全scaleの
  画像ピラミッドを構築する
- デバッグ経路では中間scaleデータを保持・出力できるよう、CPUで並列に画素変換と
  画像ピラミッド生成を行う
- 通常実行ではSSIMマップだけをreadbackする

## 実装の制約

- 同一画像/動画は同一チェックを入れずに `0.00000000` を出力されるようにする
- その他のペアは `dssim.exe` に対する相対誤差1%未満を維持する
- 上記を満たせば浮動小数点数の加乗算は結合法則を満たすとして変形してもよい
- blur weightおよびSSIM定数は変更しない

## Vulkan SDKとシェーダー

configureの前にVulkan SDKをインストールしてください。ビルドにはSDKの
loaderライブラリ、ヘッダー、`slangc` (2026.14.1+) / `spirv-val` (2026.3+)が必要で、いずれかが見つからない場合は
configureが明確なエラーで終了します。PowerShellからシェーダーコンパイラを
確認できます。

```powershell
& .\third_party\slang-2026.14.1\bin\slangc.exe -version
```

ビルド時に次のSlangコンピュートシェーダーをSPIR-Vへコンパイルします。
アプリケーション起動時のシェーダーコンパイルは行いません。

- `rgba8_to_linear.comp`
- `vulkan_yuv_to_rgba8.comp`
- `downsample_2x2.comp`
- `lab_preprocess.comp`
- `stage0_absdiff.comp`
- `stage0_score.comp`

オリジナルのdssimアルゴリズムと比較しながら実装するため、オリジナルの実装のソースコードを `src_reference/` に残しています。
