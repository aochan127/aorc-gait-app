# AORC Gait App (Shin-Splints Risk Markers)

ブラウザだけで動く **歩行フォーム簡易評価**（プロトタイプ）。  
MediaPipe Pose Landmarker を用いて、以下の指標を可視化します：

- 膝内反角（左/右）
- 骨盤ドロップ
- ステップ幅（足首距離 / 骨盤幅）
- 脛骨傾き（側方ビュー）

> 目的：**シンスプリントにつながりやすいパターンの簡易スクリーニング**

## 使い方（GitHub Pages）

1. このリポジトリを自分のアカウントに作成（空でもOK）
2. ローカルでファイルを配置して push：

```bash
git init
git checkout -b main
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin git@github.com:YOUR_GITHUB_USERNAME/aorc-gait-app.git
git push -u origin main
```

3. （推奨）Actions による Pages 自動デプロイが走ります。  
   完了すると URL は `https://YOUR_GITHUB_USERNAME.github.io/aorc-gait-app/`。

> もし Actions を使わない場合：Settings → Pages → “Deploy from a branch” → `main` / `/ (root)` を選択。

## 注意事項

- iOS/Safari では HTTPS が必要（GitHub Pages は HTTPS のため問題なし）。
- 本ツールは教育・セルフチェック目的のプロトタイプ。臨床判断と併用してください。

## ライセンス

MIT
