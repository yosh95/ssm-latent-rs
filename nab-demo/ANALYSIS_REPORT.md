# NAB Anomaly Detection — 分析レポート（JEPA復活版）

**最終更新日**: 2026-05-17
**変更**: Pure Mamba から JEPA（MultiScaleLatentPredictor）に回帰。異常検知に適した損失重みで再構成。

---

## 1. JEPA 復活の根拠

### 先行アプローチ（Pure Mamba）の構造的限界

Pure Mamba 版（ANALYSIS_REPORT 2026-05-16）は 1-step 予測誤差のみを異常シグナルとしていた。
しかし NAB の異常パターンには以下が含まれ、**1-step 予測誤差では原理的に検出できない**ものがある：

| 異常タイプ | 1-step 誤差 | 長期予測誤差 | 再構成誤差 |
|:---|:---|:---|:---|
| スパイク異常 | ✅ 大 | ✅ 大 | ✅ 大（正常 manifold から外れる） |
| レベルシフト | ❌ 小（shift 後は平坦） | ✅ 大（旧レベルに留まる） | ❌ 小（正常パターンの一部に見える） |
| トレンド変化 | ❌ ほぼゼロ | ✅ 徐々に拡大 | ❌ 小 |
| 周期性崩壊 | ❌ ほぼゼロ | ✅ 周期ズレが蓄積 | ✅ 中（周期構造の崩れ） |
| 構造的異常 | ❌ 小 | ✅ 大 | ✅ **大（正常 manifold 外）** |

JEPA の encoder/decoder が提供する**再構成誤差**は、1-step 予測誤差と**直交するシグナル**であり、
特に「構造的異常」（正常の manifold に乗っていないデータ点）の検出に不可欠。

### Circle デモで実証された JEPA の長期推論安定性

Circle World デモでは、JEPA の潜在空間閉ループ推論が 20 ステップ以上にわたって
ドリフトなしで円軌道を維持できることを実証した。同じタスクを Pure Mamba で行うと
数ステップでドリフトし、円から外れる。

この「長期閉ループ予測の安定性」は、NAB での**マルチホライズン予測誤差**による
異常検出の前提条件である。

---

## 2. 新アーキテクチャ（JEPA + Multi-Scale SSM）

```
Observation[8D] → Encoder → z[24D]
                   ↓
              Decoder → Recon_x[8D]    ← 再構成誤差シグナル（構造的異常）
                   ↓
z[24D] + zero_action[1D] → Fusion → u[24D]
                   ↓
          MultiScaleSsmBlock(3層)
            fast / medium / slow
                   ↓
              z_pred[24D]              ← 潜在予測誤差シグナル（時間的異常）
                   ↓
              Decoder → Pred_x[8D]     ← 観測予測誤差シグナル
```

| 項目 | 旧 (Pure Mamba) | 新 (JEPA復活) |
|:---|:---|:---|
| モデル | MultiScaleMambaPredictor | **MultiScaleLatentPredictor** |
| パラメータ | ~45K | ~25K (d_model=24) |
| 損失 | MSE(pred[t], target[t+1]) のみ | latent MSE + recon MSE + minimal VICReg |
| curvature weight | N/A | **0.0**（異常の曲がりを罰しない） |
| stability weight | N/A | **0.001**（崩壊防止の最小限） |
| 異常スコア | 1-step 予測誤差のみ | **再構成誤差 + 潜在予測誤差 + 観測予測誤差** |
| 訓練範囲 | 先頭15% | 先頭15%（継承） |
| 正規化 | MAD robust | MAD robust（継承） |
| 特徴量 | 8D（継承） | 8D（継承） |
| キャリブレーション | MAD + EWMA + power 0.25 | 同（継承） |

---

## 3. 損失関数

```
L = MSE_latent(z_pred[t], z[t+1])                          ← 潜在空間での予測（core JEPA）
  + recon_w · (MSE(recon_x[t], x[t]) + MSE(pred_x[t], x[t+1]))  ← 再構成品質
  + stability_w · VICReg(z[t])                              ← 潜在崩壊防止（最小限）
  + curvature_w · Σ|z_t - 2z_{t-1} + z_{t-2}|²            ← ★ 0.0（異常検知では無効化）
```

### 重要な設計判断：curvature_weight = 0.0

Temporal Straightening（曲率損失）は、Circle デモのような**予測タスク**では軌道を滑らかにする効果があるが、
**異常検知では「軌道の曲がり」こそが検出すべき異常シグナルである**。したがって完全に無効化する。

### 重要な設計判断：stability_weight = 0.001

VICReg を完全にゼロにすると、JEPA の非対照学習において潜在表現が定数に崩壊するリスクがある。
最小限の重み（0.001）で崩壊を防ぎつつ、異常の「分布外性」を過度に抑制しない。

---

## 4. 異常スコア（トリプルシグナル）

```
anomaly_score[t] = α · recon_err[t] + β · latent_pred_err[t] + γ · obs_pred_err[t]

recon_err[t]     = ||x[t] - decoder(encoder(x[t]))||²
  → 「このデータ点は正常 manifold に乗っているか？」

latent_pred_err[t] = ||z[t] - z_pred[t-1]||²
  → 「この遷移は正常ダイナミクスに従うか？（潜在空間）」

obs_pred_err[t]   = ||x[t] - x_pred[t-1]||²
  → 「この遷移は正常ダイナミクスに従うか？（観測空間）」
```

デフォルト重み: `α=1.0, β=0.5, γ=0.5`（config から調整可能）

---

## 5. ハイパーパラメータ

| パラメータ | 値 | 根拠 |
|:---|:---|:---|
| d_model | 24 | 最小限。以前 64→32 で精度変化なし、24 でも十分と予想 |
| d_state | 8 | 最小限の状態次元 |
| expand | 2 | 標準 |
| n_heads | 2 | d_model=24, expand=2 → d_inner=48, d_head=24 |
| n_layers | 3 | fast/medium/slow のマルチスケール |
| stability_weight | 0.001 | 崩壊防止の最小限 |
| curvature_weight | 0.0 | 異常検知では必須 |
| recon_weight | 1.0 | 標準 |
| α_recon | 1.0 | 再構成誤差に最大重み |
| β_latent | 0.5 | 潜在予測誤差 |
| γ_obs | 0.5 | 観測予測誤差 |

---

## 6. Quick モード用データセット

4 カテゴリから 1 つずつ選択（計 4 ファイル）:

| カテゴリ | ファイル | 選択理由 |
|:---|:---|:---|
| realKnownCause | ambient_temperature_system_failure.csv | 明確な異常、中規模 |
| realAWSCloudwatch | ec2_cpu_utilization_5f5533.csv | 実運用データ、ノイズ多 |
| artificialWithAnomaly | art_daily_jumpsdown.csv | 人工的だがレベルシフトあり |
| realTraffic | occupancy_t4013.csv | 周期性＋異常 |

`nab_config.toml` の `mode.current = "full"` で全データセット評価に切り替え可能。

---

## 7. SOTA との差を埋める次の一手

1. **マルチホライズン潜在予測**: 現在は 1-step のみ。5-step, 10-step の潜在予測誤差を追加。
   ただし、JEPA の潜在空間閉ループの安定性が前提。まずは 1-step でベースラインを確立。
2. **アンサンブル**: 3〜5 シードのモデルを訓練し、異常スコアの中央値/最大値を合成。
3. **a_re 減衰率の調整**: 長期記憶を強化するため、slow レイヤーの a_re 範囲を緩和。
