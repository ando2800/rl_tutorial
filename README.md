# 強化学習アルゴリズム比較プロジェクト

## 概要

本プロジェクトは、CartPole-v1環境における主要な強化学習アルゴリズム（REINFORCE, DQN, A2C, PPO, DDQN）の実装、学習、および性能比較を行うものです。各アルゴリズムの基本的な動作を理解し、その性能特性を評価することを目的としています。

## 特徴

*   **複数の強化学習アルゴリズム**: REINFORCE, DQN, A2C, PPO, DDQN を実装。
*   **CartPole-v1環境**: Gymnasiumライブラリを使用したCartPole-v1環境での学習。
*   **学習結果の可視化**: 各アルゴリズムの報酬履歴をプロットし、性能を比較。
*   **詳細レポート**: 各アルゴリズムの学習過程、DQN/PPOの改善に関する詳細なレポートを `docs/cartpole.md` に記述。

## 環境セットアップ

本プロジェクトを実行するには、Python 3.12.8以上が必要です。以下の手順で仮想環境をセットアップし、必要な依存ライブラリをインストールしてください。

```bash
# 既存の仮想環境を削除（初回または再構築時）
rm -rf .venv

# 新しい仮想環境を作成
python3 -m venv .venv

# 依存ライブラリをインストール
./.venv/bin/pip install -r requirements.txt

# プロットに必要なライブラリを追加でインストール
./.venv/bin/pip install seaborn matplotlib
```

## 実行方法

各アルゴリズムの学習は、`main.py` スクリプトを通じて実行できます。`--algorithm` オプションで実行したいアルゴリズムを指定してください。設定ファイルは `configs/` ディレクトリにあります。

*   **REINFORCEの学習実行**:
    ```bash
    ./.venv/bin/python main.py --algorithm REINFORCE
    ```

*   **DQNの学習実行**:
    ```bash
    ./.venv/bin/python main.py --algorithm DQN
    ```

*   **A2Cの学習実行**:
    ```bash
    ./.venv/bin/python main.py --algorithm A2C
    ```

*   **PPOの学習実行**:
    ```bash
    ./.venv/bin/python main.py --algorithm PPO
    ```

*   **DDQNの学習実行**:
    ```bash
    ./.venv/bin/python main.py --algorithm DDQN
    ```

### 結果プロットの生成

全てのアルゴリズムの学習が完了した後、以下のコマンドで報酬の比較プロットを生成できます。

```bash
./.venv/bin/python plot_rewards.py
```

## 結果の確認

*   **学習ログ**: 各アルゴリズムの学習ログは `logs/` ディレクトリにCSV形式で保存されます（例: `logs/dqn_cartpole_rewards.csv`）。
*   **比較プロット**: 生成された報酬の比較プロットは `logs/all_rewards_comparison.png` に保存されます。
*   **詳細レポート**: 各アルゴリズムの学習過程、DQN/PPOの改善に関する詳細な考察については、`docs/cartpole.md` を参照してください。

## 今後の展望

*   より複雑な環境でのアルゴリズムの評価。
*   ハイパーパラメータ自動探索ツールの導入。
*   その他の強化学習アルゴリズム（例: SAC, DDPG, TD3）の実装と評価。
