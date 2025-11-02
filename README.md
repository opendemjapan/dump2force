# dump2force

 本リポジトリは LIGGGHTS の pair/local スタイルの dump を VTK/VTU の線分ネットワークに変換し, Louvain 法に基づくコミュニティ属性を付与するツール `dump2force.py` を提供する. 

## 特徴

 - VTU/VTK の両形式を出力可能である (`--vtk-format {vtu,vtk}`, `--encoding {ascii,binary}`). 
 - PVD は出力しない. 
 - 出力先は既存ディレクトリのみである (`--outdir` は既存ディレクトリを指定). 
 - VTK legacy 出力時には NaN/Inf を `--nan-fill` の値で置換する. 
 - `ITEM: ENTRIES ...` から列名を自動解釈し, 必須 12 列を要求する. 必須列は `x1 y1 z1 x2 y2 z2 id1 id2 periodic fx fy fz` である. 
 - 13 列目以降の任意スカラー列と, 連続 3 列から成るベクトル列 (`*x/*y/*z`, `*1/*2/*3`, `*[1]/[2]/[3]`) をパススルーする. 
 - Louvain 法により無向グラフでコミュニティ検出を行う. ノードは粒子 ID, エッジ重みは接触力の大きさである. `--seed`, `--resolution` で制御可能である. 
 - CellData として `force`, `connectionLength` に加えて, 任意スカラー/ベクトル, `community`, `intra_comm`, `comm_mean_*`, `comm_sum_*` を出力する. 
 - `--write-pointdata` 指定時は PointData として `node_community`, `node_degree`, `node_force_sum` を出力する. 

## 依存関係

 - Python 3.8 以降を推奨する. 
 - NumPy. 
 - NetworkX. 
 - python-louvain (`import community as community_louvain`). 

```bash
pip install numpy networkx python-louvain
```
 
## 使い方

```bash
python dump2force.py DUMPFILE   --vtk-format {vtu,vtk}   --encoding {ascii,binary}   [--keep-periodic]   [--resolution 1.0]   [--seed 42]   [--write-pointdata]   [--outdir EXISTING_DIR]   [--nan-fill 0.0]   [--quiet]
```
 
### 主な引数

 - `dumpfile`: 入力 dump ファイルである. `.gz` も可である. 
 - `--vtk-format`: `vtu` は XML UnstructuredGrid, `vtk` は legacy POLYDATA である. 既定は `vtu` である. 
 - `--encoding`: `ascii` または `binary`. 既定は `ascii` である. 
 - `--keep-periodic`: 周期境界の接触を保持する. 指定しない場合は除外する. 
 - `--resolution`: Louvain 法の resolution パラメータである. 既定は 1.0 である. 
 - `--seed`: Louvain 法の乱数 seed である. 既定は 42 である. 
 - `--write-pointdata`: ノード属性を PointData に書き出す. 
 - `--outdir`: 既存の出力先ディレクトリである. 自動作成は行わない. 既定は入力ファイルと同じディレクトリである. 
 - `--nan-fill`: legacy VTK で NaN/Inf を置換する値である. 既定は 0.0 である. 
 - `--quiet`: 進捗出力を抑制する. 
 
### 入力列仕様

 必須 12 列は `x1 y1 z1 x2 y2 z2 id1 id2 periodic fx fy fz` である. 
 13 列目以降はスカラー列をそのまま通し, 3 列から成るベクトル列を自動認識する. ベクトルは `suffix = x/y/z`, `1/2/3`, `[1]/[2]/[3]` を許容する. 

### 出力

 - VTU (`ascii/binary`): UnstructuredGrid の Points/Cells と PointData/CellData を書き出す. 
 - VTK legacy (`ascii/binary`): POLYDATA の POINTS と LINES を書き出す. NaN/Inf は `--nan-fill` に置換する. 
 - CellData: `force`, `connectionLength`, 任意スカラー/ベクトル, `community`, `intra_comm`, `comm_mean_*`, `comm_sum_*`. 
 - PointData (`--write-pointdata` 指定時): `node_community`, `node_degree`, `node_force_sum`. 

### 例

```bash
# 既存ディレクトリ ./out に VTU (ASCII) を書き出す
mkdir -p out
python dump2force.py dump.pairlocal --vtk-format vtu --encoding ascii --outdir ./out --write-pointdata

# legacy VTK (BINARY) で NaN/Inf を 0.0 に置換して書き出す
python dump2force.py dump.pairlocal --vtk-format vtk --encoding binary --nan-fill 0.0
```

## 設計と実装メモ

 - 解析器は Pizza.py の `bdump` の最小限機能を Python 3 用に再実装したものである. `.gz` を含む逐次読み出しに対応する. 
 - Louvain 法は `networkx` と `python-louvain` を用いて実装する. エッジ重みは `||(fx,fy,fz)||` である. 
 - VTK legacy では ParaView が NaN/Inf を属性として解釈できず, 警告を出す場合があるため, 書き出し前に置換する. 

## 制限事項

 - 入力 dump に重複した timestep が含まれている場合, 重複は自動的に除去する. 
 - 新規ディレクトリの自動作成は行わない. 既存ディレクトリを指定すること. 
 - `python-louvain` と `networkx` が未導入の場合, コミュニティ検出は実行できない. 

## ライセンスと出典

 本リポジトリは Pizza.py および LPP に依拠するため, GNU GPL v2 に適合する `LICENCE` を含める. 
 Pizza.py は GNU GPL v2 で配布されている. 
 LPP は Pizza.py を LIGGGHTS 用に修正した派生物であり, 同様に GNU GPL v2 で配布されている. 
 本ツール全体を GNU GPL v2 で配布する. 

 - Pizza.py. Sandia National Laboratories. GNU GPL v2. 
 - LPP (LIGGGHTS Post-Processing). GNU GPL v2. 

## 謝辞

 Pizza.py と LPP の開発者に謝意を表する. なお, 本リポジトリは Sandia National Laboratories および DCS Computing/CFDEM®project の公式配布物ではない. 
