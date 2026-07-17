#!/bin/bash
DIR="$1"

# 1) Git リポジトリか？
if git -C "$DIR" rev-parse --is-inside-work-tree &>/dev/null; then
  # 2) origin URL を取得
  url=$(git -C "$DIR" config --get remote.origin.url)
  if [[ "$url" == *github.com* ]]; then
    echo "✅ $DIR は GitHub リポジトリです: $url"
    exit 0
  else
    echo "⚠️  Git 管理下ですが、GitHub リポジトリではありません（origin: $url）"
    exit 1
  fi
else
  echo "❌ $DIR は Git リポジトリではありません"
  exit 1
fi
