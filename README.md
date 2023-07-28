# StableEmotion
<img width="260" alt="image" src="https://github.com/CeitherNSW/StableEmotion/assets/128119791/56f573ec-42b9-49e7-a8e1-280cb5ee8265">

按照图片所示结构存放train及valid图片，off test内容在仓库中

  ###   当前目标：
  *   ~~需寻找一种合适的模型将图片处理为透明背景png~~
  *   ~~确定使用传统CNN或深度学习模型进行分类~~
    
  * 今日上传annotation.json文件，需进行进一步操作
    * 需将annotation区块中的image_id与images区块中的id对应
    * 为每个images.filename创建同名的txt文档
    * 文档中按
    ``annotations.category_id(空格)annotations.segmentation``
    的格式保存信息

    * 若已有相同名字的txt文档则跳过
    * 具体样式参考
    ``<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>``
    其中x1,y1,x2,y2...xn,yn为``annotations.segmentation``中的各个元素除以640后的float值。