# people_detection_ros2
![Shigure Plugin](https://img.shields.io/badge/shigure-plugin-blue)

人物の推定をするROS2パッケージ

## 動作要件
* Python 3.6 以上
* Ubuntu 20.04 LTS
* ROS2 foxy
* [yolact](https://github.com/dbolya/yolact)

## 動作確認環境
* Python 3.8.12 以上
* Ubuntu 20.04 LTS
* ROS2 foxy
* [yolact](https://github.com/dbolya/yolact)
  * branch : master 
  * commit hash : [57b8f2d95e62e2e649b382f516ab41f949b57239](https://github.com/dbolya/yolact/commit/57b8f2d95e62e2e649b382f516ab41f949b57239)

## インストール
1. 任意の場所で `mkdir yoloct` を実行します。
1. `cd yolact`
1. `git clone https://github.com/dbolya/yolact.git` を実行し [yolact](https://github.com/dbolya/yolact) を git から clone します。
1. `cd yolact && touch __init__.py` を実行します。
1. `echo "export PYTHONPATH="$(pwd)":$PYTHONPATH" >> ~/.bashrc` を実行し yolact を PYTHONPATH に通します。
1. このリポジトリをros2ワークスペースにクローンします(大体は `~/ros2_ws/src` )。
1. ros2ワークスペースのルートに移動します(大体は `~/ros2_ws` )。
1. `cd src/people_detection_ros2/people_detection_ros2`
1. `python3 -c "from people_detection_wrapper import PeopleDetectionWrapper"` を実行して、エラーを解消します。
   * 大体は `from data.config import cfg, mask_type` を `from yolact.data.config import cfg, mask_type` といった先頭に `yolact` をつけるだけで解決します。
1. `colcon build --base-paths src/people_detection_ros2` を実行しパッケージをビルドします。
1. `. install/setup.bash` を実行してパッケージをアップデートします。

## params.ymlでノードを起動する
1. ros2ワークスペースのルートに移動します。
1. `cp ./src/people_detection_ros2/people_detection_ros2/params.yml.sample ./src/people_detection_ros2/people_detection_ros2/params.yml` を実行してファイルをコピーします。
1. `./src/people_detection_ros2/people_detection_ros2/params.yml` を開き編集します。パラメータについては「ROS2パラメータの一覧」を参照してください。
1. `ros2 run people_detection_ros2 people_detection_ros2 __params:=src/people_detection_ros2/people_detection_ros2/params.yml` を実行します。

## ノードの一覧
* /openpose_node

## トピックの一覧
* /people_detection \[sensor_msgs/msg/Image\]

## ROS2パラメータの一覧
* is_debug_mode : trueの場合、デバッグモードで実行します。 
* trained_model_path : 学習済みモデルのディレクトリを指定します。
* score_threshold : 推定スコアの下限値を指定します(0.0 - 1.0)。
* is_image_compressed : trueの場合、入力される画像はCompressedImage型を期待します。falseの場合は、入力される画像はImage型を期待します。
* image_node : 入力画像のノードを指定します。

## ライセンス
MIT License
