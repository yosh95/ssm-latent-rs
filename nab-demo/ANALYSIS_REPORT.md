# NAB Anomaly Detection — SSM+Latent 深入分析レポート

**日付**: 2026-05-12  
**対象**: `nab-demo/` 以下のSSM+Latent異常検知パイプライン  
**SOTA（NABベンチマーク）**: ARTime Standard=57.66 / Numenta Standard=46.63  
**現在スコア**: ssm_latent Standard=**-27.95**（SOTAとの差: **85.6ポイント**）

---

## 1. 現在のスコア（公式NAB評価）

`config/thresholds.json` に記録された公式スコア:

| Profile | ssm_latent | ARTime (SOTA) | Numenta | contextOSE | Null baseline |
|:---|:---|:---|:---|:---|:---|
| **Standard** | **-27.95** | **57.66** | 46.63 | 46.17 | -116.00 |
| **Reward Low FP** | **-64.90** | **35.16** | 30.43 | 39.39 | -116.00 |
| **Reward Low FN** | **-68.27** | **47.66** | 26.63 | 22.67 | -232.00 |

> ⚠️ `BENCHMARK_REPORT.md` には Standard=37.95 と記載されているが、これは公式NABスコアリング
> スクリプト（`evaluate_nab.py` / `nab/sweeper.py`）の結果ではなく、
> 内部的な評価ロジックによるものと推定される。公式スコアとは65.9ポイント乖離している。

---

## 2. 根本原因分析（5層モデル）

```
Layer 5: スコアリング（sigmoid変換）──→ スコア分布が[0, 0.01]に押しつぶし
                                      └──→ NAB閾値最適化が事実上不可能
Layer 4: EMAパラメータ設定 ──────────→ 初期値0が異常スパイクを作出
                                      └──→ 異常汚染(contamination)なし
Layer 3: 推論パイプライン ───────────→ チャンク境界で状態リセット→偽スパイク
Layer 2: 学習パイプライン ───────────→ データ15%, 50epoch, 固定LR=1e-3
Layer 1: モデルアーキテクチャ ────────→ 単層SSM, action_dim無駄, d_model=64
```

各レイヤーの問題が複合的に作用し、特にLayer 4-5がスコアに対して支配的です。

---

## 3. 詳細分析

### 3.1 🔴 Layer 5: Sigmoid変換がスコアを潰している（最も致命的）

**現状コード** (`main.rs` 148行目):
```rust
let s = 1.0 / (1.0 + (-(z_score - 4.0)).exp());
```

**問題**: `sigmoid(z - 4)` は `z < 4` のとき ほぼ 0 を返す。
典型的なz-scoreの分布（0〜3の範囲に95%以上が収まる）では、
スコアの99%以上が ≈0.02 以下に押しつぶされる。

**結果**: 
- NABの閾値最適化（`sweeper.py`）は `anomaly_score` が [0, 1] に分散していることを前提とする
- スコアがほぼ全面的に ≈0 の場合、最適な閾値は ≈0.999648 になる
- この閾値を超えるスパイクは、チャンク境界エラー（Layer 3）や初期EMA過渡応答（Layer 4）による偽陽性
- したがって、TPはほぼなく、FPでスコアが大幅に減点される → **負のスコア**

### 3.2 🔴 Layer 4: EMAの初期値と汚染問題

**現状コード** (`main.rs` 137-143行目):
```rust
let mut moving_mean = 0.0f32;   // ← ゼロ初期化
let mut moving_var = 0.05f32;   // ← 実分散より過小
let alpha = 0.05;
```

**問題3点**:
1. **初期値ゼロ**: 最初のステップの誤差が`moving_mean=0`から大きく離れ、z_scoreが不当に高くなる
2. **分散過小**: `0.05` は実際の誤差分散より遥かに小さい（1桁以上小さいことが多い）
3. **汚染(contamination)**: 異常値を含むステップでもEMAを更新するため、異常後のベースラインが歪む

### 3.3 🟡 Layer 3: チャンク境界でSSM状態がリセット

**現状コード** (`main.rs` 107-122行目):
```rust
let chunk_size = 5000;
for start in (0..seq_len).step_by(chunk_size) {
    let chunk_tensor = ...;
    let (z, pred_z, reconstructed, _) = model.forward(chunk_tensor, ...);
    // ← 各チャンクで状態をリセット！
}
```

**問題**: `forward()`（parallel scan）を使うため、各チャンクは独立した系列として処理される。
5000ステップごとに隠れ状態がゼロリセットされ、境界で必ず予測誤差の急上昇が発生する。

**修正**: `forward_step()` を使ったストリーミング推論にすべき。（`LatentState` + `step()` APIが既に存在する）

### 3.4 🟡 Layer 2: 学習が不十分

| パラメータ | 現状 | 推奨 |
|:---|:---|:---|
| 学習データ | 先頭15% | フルデータ（自己回帰的） |
| エポック数 | 50 | 200-500 |
| 学習率 | 1e-3 固定 | Cosine schedule, peak=3e-4 |
| Loss重み | stab=1.0 / curv=0.5 / recon=2.0 | stab=0.1 / curv=0.05 / recon=5.0 |
| Early stopping | なし | patience=20 |

### 3.5 🟠 Layer 1: モデルアーキテクチャ

| パラメータ | 現状 | 推奨（Phase 3） |
|:---|:---|:---|
| SSM層数 | 1 | 2-3（マルチスケール） |
| d_model | 64 | 128-256 |
| d_state | 16 | 32-64 |
| action_dim | 1（無駄） | 0（削除）または1（time feature） |
| 入力特徴量 | 生の正規化値1つ | 窓統計量（mean, stdを追加） |

---

## 4. 推定インパクト

| 改善Phase | 内容 | 推定Standard改善 | 累積スコア |
|:---|:---|:---|:---|
| 現在 | — | — | -27.95 |
| Phase 1 | スコアリング修正（MAD+EWMA, sigmoid除去） | +30〜50 | +2〜+22 |
| Phase 2 | ストリーミング推論（状態継続） | +10〜20 | +12〜+42 |
| Phase 3 | 学習改善（LR schedule, loss重み, データ増） | +5〜15 | +17〜+57 |
| Phase 4 | マルチスケールSSM + 入力特徴量エンジニアリング | +5〜15 | +22〜+72 |

Phase 1 だけで SOTA の約30-40% に到達し、全Phase適用で SOTA の 40-130% の範囲到達が見込めます。

---

## 5. BENCHMARK_REPORT.md の問題点

| 項目 | BENCHMARK_REPORT.md | 実際（thresholds.json） |
|:---|:---|:---|
| Standard | 37.95 | **-27.95** |
| Reward Low FP | 22.03 | **-64.90** |
| Reward Low FN | 47.05 | **-68.27** |

**原因**: `BENCHMARK_REPORT.md` は単独ファイルのスコアを合計している可能性があるが、
`thresholds.json` のスコアは NAB公式スコアリング（`sweeper.py` + プロファイル重み付け + 
閾値最適化）を経た正規化済みスコア。独自評価と公式評価のスケールが異なる。

---

## 6. 具体的な修正コード

以下は Phase 1 + Phase 2 を同時に実装した修正版 `main.rs` です。

主な変更点:
1. **スコアリング**: MAD（Median Absolute Deviation）キャリブレーション + EWMA汚染防止 + sigmoid変換の廃止
2. **推論**: `forward_step()` を使ったストリーミング推論（状態継続）
3. **学習**: より長いデータ使用 + エポック増加 + Loss重み調整
4. **モデル**: action_dim=0 対応（不要なパラメータ削減）