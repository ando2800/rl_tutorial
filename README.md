# DQN CartPole プロジェクト

このプロジェクトは、GymnasiumのCartPole-v1環境を解決するためのDeep Q-Network (DQN) エージェントを実装しています。

## 実装内容

- Qネットワーク、リプレイバッファ、ターゲットネットワークを備えたDQNエージェント。
- 指定されたエピソード数で実行されるトレーニングループ。
- YAMLファイルによる設定管理。
- エピソードごとの報酬をCSVファイルに記録。
- 学習済みモデルの保存。
- 依存関係を管理するための仮想環境 (`.venv`)。

## ディレクトリ構造

```
.
├── agents
│   └── dqn.py            # DQNTrainer クラスの実装
├── configs
│   └── dqn_cartpole.yaml # トレーニング実行用の設定ファイル
├── logs
│   └── dqn_cartpole_rewards.csv # エピソードごとの報酬を記録するCSVファイル
├── models
│   └── dqn_cartpole.pt   # 保存されたモデルのstate dictionary
├── .venv/                   # Python仮想環境
├── main.py                 # トレーニングを実行するメインスクリプト
└── README.md               # このファイル
```

## 実行方法

1.  **環境のセットアップ:**

    まず、仮想環境を作成し、アクティベートします。必要な依存関係はすでにこの環境内にインストールされています。

    ```bash
    # 仮想環境を作成します (まだ作成していない場合)
    python3 -m venv .venv

    # 仮想環境をアクティベートします
    source .venv/bin/activate
    ```

2.  **依存関係のインストール:**

    環境をゼロからセットアップする必要がある場合は、pipを使用して必要なパッケージをインストールします。
    ```bash
    pip install -r requirements.txt
    ```
    *(注: 便宜上、`requirements.txt` ファイルが生成されます。)*

3.  **トレーニングの実行:**

    メインスクリプトを実行してDQNエージェントのトレーニングを開始します。
    ```bash
    python3 main.py
    ```

## 現在のステータス

DQNエージェントのトレーニングが成功し、CartPole-v1環境を解決しました！最後の100エピソードの平均スコアは195.0を超え、安定したパフォーマンスを示しています。

- 報酬の学習曲線は `logs/rewards_plot.png` に保存されています。
- 学習済みモデルが環境をプレイする動画は `logs/videos/` ディレクトリに保存されています。