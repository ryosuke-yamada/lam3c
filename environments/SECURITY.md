# コンテナ再ビルド時のセキュリティ確認（2026-10-25）

今回の目的は、ローカルで適用したBrotli修正が再ビルドで失われないようにすること。
ホスト設定、認証情報、既存コンテナ、依存ロックは変更しない。既存タグ・成果物は
バックアップとして保持し、新しいタグで構築・検証してから利用先を切り替える。
この変更だけで既存コンテナや配布済みイメージが更新されるわけではない。

根拠は[Node.js公式告知](https://nodejs.org/en/blog/vulnerability/october-2026-security-releases)と
[Debian公式追跡情報](https://security-tracker.debian.org/tracker/CVE-2026-111497)。
Debianは修正をバックポートするため、Brotliの上流バージョン番号だけでは判断しない。
以下は2026-10-25時点の基準であり、将来の全脆弱性への安全性を保証しない。

Webサイト用gh-pagesブランチのDockerfileと開発・CI用Composeを対象とする。
Node 22.23.4 bookwormの同一公式digestへ固定し、yarn.lockは維持する。
依存インストールに失敗しても成功扱いにする旧シェル式を修正した。

```sh
sh environments/check-container-security.sh IMAGE
```

IMAGEは新しく構築したイメージ名。Nodeの修正版判定、Brotli往復処理と修正に対応する
エラー定数をネットワークなしで検査する。
ローカル検証: linux/arm64で依存インストールを含むビルド、`yarn lint`、`yarn build`成功。
公開サイトへのデプロイは未実施。mainの研究コードは今回の対象外。


検査は先に作成し、修正前イメージでは未修正Node/システムBrotliまたは旧Playwright起動経路を
理由に失敗（RED）、修正後のイメージでは成功（GREEN）した。
ShellCheckも実施。GitHub CIの成功やデプロイの成功を意味するものではない。
Python wheel等に独自に内包されたライブラリまですべて修正したという主張ではない。

再ビルドしたNodeでnodejs.orgへの正常TLS接続と、既知の問題証明書を提示する
検査専用ホスト2経路の拒否を確認した。案内ページ自体の接続可否とは区別する。
