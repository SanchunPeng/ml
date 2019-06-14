## 1、简介

2018年3月30日，TensorFlow团队展示了他们著名的机器学习（ML）框架TensorFlow的Javascript版本。

TensorFlow.js允许导入预训练的模型，可以重新训练现有的导入模型，以及导出模型。

ml5.js构建于Tensorflow.js之上，是一个友好的高级界面，可以访问浏览器中的机器学习算法和模型。利用浏览器内置的GPU加速计算。

## 2、ML5能实现的功能以及使用场景
- Image
![image](https://sanchunpeng.github.io/ml/images/todoimage.jpg)
图像识别，人体姿态检测，人体部位分割，图像分割，样式转换，成对的图像转换，条件变分自编码器，深度卷积对抗生成网络，自动完成涂鸦，实时物体检测

- Sound  
![image](https://sanchunpeng.github.io/ml/images/todosound.jpg)   
音频识别，基因检测

- Text   
![image](https://sanchunpeng.github.io/ml/images/todotext.jpg)
文本生成，情感检测，语义化

- Helpers   
![image](https://sanchunpeng.github.io/ml/images/todohelper.jpg)   
特征提取，K近邻分类

### 2.1、图像识别imageClassifier
ml5.imageClassifier()是一种使用预训练模型对图像进行分类的对象的方法。

ml5提供了一些训练好的模型，是在大约1500万个图像（ImageNet）的数据库上进行训练的。ml5中可用的模型有'MobileNet', 'Darknet' and 'Darknet-tiny',当然也可以是任何训练好的图像分类模型

ml5.imageClassifier()返回一个数组，包含识别出的结果以及置信度

```
ml5.imageClassifier(model)
ml5.imageClassifier(model, ?callback)
ml5.imageClassifier(model, ?options, ?callback)
ml5.imageClassifier(model, ?video, ?options, ?callback)
```

```
.classify(input, ?callback)
```
![image](https://sanchunpeng.github.io/ml/images/class1.jpg)

也可以识别多个图片   
![image](https://sanchunpeng.github.io/ml/images/class2.jpg)


### 2.2、人体姿态检测poseNet
ml5.poseNet()用于实时人体姿态检测，可以检测图像/视频中的一个人，也可以检测图像/视频中的多个人。主要用于检测图像或者视频中的人的关键身体关节的位置。   

![image](https://sanchunpeng.github.io/ml/images/runner.jpg)


![image](https://sanchunpeng.github.io/ml/images/runpose.jpg)

不仅可以检测一个人也可以检测多个人：
![image](https://sanchunpeng.github.io/ml/images/singlehuman.gif)


![image](https://sanchunpeng.github.io/ml/images/multihuman.gif)


posenet将返回一个pose对象，该对象包含每个被检测者的关键点(关节点)列表和置信度。

```
ml5.poseNet(?video, ?type, ?callback)
ml5.poseNet(?video, ?options, ?callback)
ml5.poseNet(?callback, ?options)

```
```
.singlePose(?input)  
.multiPose(?input)
```


### 2.3、人体部位分割bodyPix
ml5.bodyPix()用于对人物及身体部位进行分割，也可以说是分类，将图像中的像素分为两类，第一类：人，可以进一步可以分为二十四个身体部位（如左脸、右脸、左手、右前腿、或背部躯干），第二类：背景
```
ml5.bodyPix(?video, ?options, ?callback)
```

```
.segment(?input, ?options, callback);
```
![image](https://sanchunpeng.github.io/ml/images/bodypix.jpg)

### 2.4、图像分割UNET
图像分割就是将图像分割成不同的部分，可以得到物体的一个准确的轮廓，其解决办法是转化为像素级的分类，将其分为物体和背景。
![image](https://sanchunpeng.github.io/ml/images/unet.jpg)  
![image](https://sanchunpeng.github.io/ml/images/unet2.jpg)
```
ml5.uNet(model)
ml5.uNet(model, ?callback)
```
```
.segment(video, ?callback);
```
图像分割在从自动驾驶汽车到卫星的许多领域都很有用。也许其中最重要的是医学影像。医学图像的微妙之处是相当复杂的。一台能够理解这些细微差别并识别出必要区域的机器，可以对医疗保健产生深远的影响。

### 2.5、样式转换styleTransfer
ml5.styleTransfer()用于将一个图像的样式转换为另一个，这是一个两步的过程，首先需要对一个特定样式的模型进行训练，然后可以将此样式应用到另一个图像。

```
ml5.styleTransfer(model, ?callback)
ml5.styleTransfer(model, ?video, ?callback)
```

```
.transfer(?callback)
.transfer(input, ?callback)
```
![image](https://sanchunpeng.github.io/ml/images/styletrans.jpg)

![image](https://sanchunpeng.github.io/ml/images/styletrans1.jpg)

![image](https://sanchunpeng.github.io/ml/images/styletrans2.jpg)

### 2.6、成对的图像转换pix2pix
通过训练成对的图像找到对应关系模型，当输入一张图片时，通过对应关系输出相应图片。

```
ml5.pix2pix(model, ?callback);
```

```
.transfer(canvas, ?callback)
```
![image](https://sanchunpeng.github.io/ml/images/pix2pix.jpg)

PS：和styleTransfer的区别是，styleTransfer训练的是要转特定样式的目标图片，pix2pix训练的是成对的图片，学习如何将输入图像映射到输出图像。



### 2.7、条件变分自编码器CVAE
自动编码器是一种神经网络，能够创建输入数据的稀疏表示，因此可以用于图像压缩。在学习了这些稀疏表示之后，有一些去噪自动编码器可以用噪声图像来表示。更妙的是，一种称为变分自动编码器的变体不仅可以学习这些稀疏表示，还可以绘制新图像。

```
ml5.CVAE(?model, ?callback)
```

```
.generate(label, callback);
```

简单的来说就是可以按照指定的类别生成样本，让机器学习图片时加入标签因素，这样就可以按照指定的标签生成图片。  
![image](https://sanchunpeng.github.io/ml/images/cave.jpg)   

![image](https://sanchunpeng.github.io/ml/images/cave3.jpg)  

![image](https://sanchunpeng.github.io/ml/images/cave2.jpg)   
可以看到，使用原图生成的自编码数据还会带有一些原图的样子，而以标签生成的解码数据，已经彻底学会了数据的分布，并生成截然不同却带有相同意义的数据。


### 2.8、深度卷积对抗生成网络DCGAN(Deep Convolutional Generative Adversarial Networks )

DCGAN包括两个网络，一个生成网络，一个判别网络，生成网络负责生成图片，判别网络判别输出是这张图像为真实图像的概率，在训练过程中，生成网络G的目标是尽量生成真实的图片去欺骗判别网络，而判别网的目标是尽量把生成的图片和真实的图片区分开来，这样就是对抗，最理想的结果是生成网络可以生成足以“以假乱真”的图片，判别网络难以判定生成网络生成的图片究竟是不是真实的，这样得到的生成式的模型，就可以用来生成图片。

```
ml5.DCGAN(?modelPath, ?callback)
```

```
.generate(callback);
```
![image](https://sanchunpeng.github.io/ml/images/dcgan.jpg)

用代表“露出笑容的女性”的z，减去“女性”，再加上“男性”，最后得到了“露出笑容的男性”

### 2.9、自动完成涂鸦SketchRNN
通过记录数以百万记的用户绘制的涂鸦，它不仅记录最终的图像，还记录绘制过程中每笔笔触的顺序和方向，而且不是简单的copy，复制的是概念

```
ml5.SketchRNN(model, ?callback)
```

```
.reset()
.generate(?seed, ?options, ?callback)
```
##### ML5库包含受支持的SketchRNN模型列表：
```
const models = [
  'alarm_clock',
  'ambulance',
  'angel',
  'ant',
  'antyoga',
  'backpack',
  'barn',
  'basket',
  'bear',
  'bee',
  'beeflower',
  'bicycle',
  'bird',
  'book',
  'brain',
  'bridge',
  'bulldozer',
  'bus',
  'butterfly',
  'cactus',
  'calendar',
  'castle',
  'cat',
  'catbus',
  'catpig',
  'chair',
  'couch',
  'crab',
  'crabchair',
  'crabrabbitfacepig',
  'cruise_ship',
  'diving_board',
  'dog',
  'dogbunny',
  'dolphin',
  'duck',
  'elephant',
  'elephantpig',
  'eye',
  'face',
  'fan',
  'fire_hydrant',
  'firetruck',
  'flamingo',
  'flower',
  'floweryoga',
  'frog',
  'frogsofa',
  'garden',
  'hand',
  'hedgeberry',
  'hedgehog',
  'helicopter',
  'kangaroo',
  'key',
  'lantern',
  'lighthouse',
  'lion',
  'lionsheep',
  'lobster',
  'map',
  'mermaid',
  'monapassport',
  'monkey',
  'mosquito',
  'octopus',
  'owl',
  'paintbrush',
  'palm_tree',
  'parrot',
  'passport',
  'peas',
  'penguin',
  'pig',
  'pigsheep',
  'pineapple',
  'pool',
  'postcard',
  'power_outlet',
  'rabbit',
  'rabbitturtle',
  'radio',
  'radioface',
  'rain',
  'rhinoceros',
  'rifle',
  'roller_coaster',
  'sandwich',
  'scorpion',
  'sea_turtle',
  'sheep',
  'skull',
  'snail',
  'snowflake',
  'speedboat',
  'spider',
  'squirrel',
  'steak',
  'stove',
  'strawberry',
  'swan',
  'swing_set',
  'the_mona_lisa',
  'tiger',
  'toothbrush',
  'toothpaste',
  'tractor',
  'trombone',
  'truck',
  'whale',
  'windmill',
  'yoga',
  'yogabicycle',
  'everything',
];
```
![image](https://sanchunpeng.github.io/ml/images/rnn.mov)


![image](https://sanchunpeng.github.io/ml/images/rnn.jpg)


### 2.10、实时物体检测YOLO
核心思想就是利用整张图作为网络的输入，直接在输出层回归 bounding box（边界框）的位置及其所属的类别。

将一幅图像分成 SxS 个网格（grid cell），如果某个 object 的中心落在这个网格中，则这个网格就负责预测这个 object。
```
ml5.YOLO();
ml5.YOLO(video);
ml5.YOLO(video, ?options, ?callback)
ml5.YOLO(?options, ?callback)
```

```
.detect(input, ?callback)
.detect(?callback)
```
![image](https://sanchunpeng.github.io/ml/images/yolo.jpg)

### 2.11 音频识别soundClassifier
通过已训练好的模型，可以检测到是否发出了某种噪音（如拍手声或哨子）或是否说了某个词（如上、下、是、否）
比如SpeechCommands18w模型，可以识别“0”到“9”、“up”、“down”、“left”、“right”、“go”、“stop”、“yes”、“no”
```
ml5.soundClassifier(?model, ?options, ?callback)
```

```
.classify(callback);
```

### 2.12 基音检测pitchDetection
基音检测算法是一种估计音频信号基音或基音频率的方法。基因检测在有调语音辩意、低速语音编码、说话人识别等方面具有关键作用。基因检测的最终目标是找到和声带震动频率一致的基因周期变化轨迹曲线，如若不能，找出尽量相吻合的轨迹曲线。   
ML5.JS的pitchDetection方法利用预先训练的机器学习基音检测模型来估计声音文件的基音。目前只支持CREPE模型。该型号是 github.com/marl/crepe 的直接端口，仅适用于浏览器麦克风的直接输入。

```
ml5.pitchDetection(model, audioContext, stream, callback);
```

```
.getPitch(?callback)
```

### 2.13 文本生成CharRNN
通过训练好的模型，通过输入文本序列，生成一个文本序列

```
ml5.charRNN(model, ?callback)
```
```
.generate(options, ?callback)
  options：{
         seed: 'The meaning of pizza is'
         length: 20,
         temperature: 0.5
        }
.feed(seed, ?callback)
.predict(temperature, ?callback)
.reset()
```
使用ml5已经训练好的模型woolf，设置length和temperatue

![image](https://sanchunpeng.github.io/ml/images/charrnn.jpg)

### 2.14 情感预测Sentiment
当输入一串文本，判断该文本的情感，负面情感或者正面情感

```
ml5.sentiment( 'moviereviews', callback )
```

```
.predict(text);
```
ml5目前支持了一个电影评论的情感分析moviereviews，对于很长的评论截取了最多200个单词，而且去除了一些生僻次，输出0表示强烈的负面情感，1表示强烈的正面情感
![image](https://sanchunpeng.github.io/ml/images/sentiment.jpg)

### 2.15 语义化word2vec
单词word转换成向量vector来表示，通过词向量来表征语义信息。弥补浅层学习特征高维稀疏和语义丢失的问题。

在常见的自然语言处理系统中，单词的编码是任意的，因此无法向系统提供各个符号之间可能存在关系的有用信息，还会带来数据稀疏问题。使用向量对词进行表示可以克服其中的一些障碍。 

是从大量文本语料中以无监督的方式学习语义知识的一种模型。简单来说可以学习出语句表含的含义，而不是单词生硬的拼凑。 

```
Word2Vec(model, ?callback)

```

### 2.15 特征提取featureExtractor
图像分类是在一个大数据集上训练出的模型来将图像分类为固定的类别集。而其中就有一个过程是特征提取，这个过程中训练出的模型可以和其他新的数据集一起训练从而得到新的分类模型，这样可以大大减少训练时间。
```
ml5.featureExtractor(model, ?callback)
ml5.featureExtractor(model, ?options, ?callback)
```


### 2.16 K近邻分离器KNNClassifier
简单的理解为由那离自己最近的K个点来投票决定待分类数据归为哪一类，与库中其他几个分类器不同的是他使用其他模型的输出或任何其他可以分类的数据构建一个KNN模型
```
ml5.KNNClassifier();
```

```
.classify(input, callback?)
.classify(input, k?, callback?)

```

## 3、使用p5.js有什么好处，是否使用有什么区别

http://alpha.editor.p5js.org/   
https://p5js.org/reference/   
```
preload()
setup()
draw()
remove()
disableFriendlyErrors
noLoop()
loop()
push()
pop()
redraw()
```
p5.js程序总是从setup()开始执行，执行过一次setup()后，便进入draw()函数的不断循环调用。
优势：提供了一整套绘图API而且可以操作DOM，比我们自己操作DOM、绘图方便多了

##### 不使用p5.js
```
// Initialize the Image Classifier method with MobileNet
const classifier = ml5.imageClassifier('MobileNet', modelLoaded);

// When the model is loaded
function modelLoaded() {
  console.log('Model Loaded!');
}

// Make a prediction with a selected image
classifier.classify(document.getElementById('image'), function(err, results) {
  console.log(results);
});
```

##### 使用p5.js

```
// Initialize the Image Classifier method with MobileNet. A callback needs to be passed.
let classifier;
// A variable to hold the image we want to classify
let img;
function preload() {
  classifier = ml5.imageClassifier('MobileNet', modelReady);
  img = loadImage('images/bird.jpg');
}
function modelReady(){
  console.log('Model Loaded!');
}
function setup() {  //setup函数，每次程序开始运行时执行一次，用于初始化。 
  createCanvas(400, 400);
  classifier.classify(img, gotResult);
  image(img, 0, 0);
}
// A function to run when we get any errors and the results
function gotResult(error, results) {
  // Display error in the console
  if (error) {
    console.error(error);
  }
  // The results are in an array ordered by confidence.
  console.log(results);
  createDiv("Label:" + results[0].label);  //用于创建DOM
  createDiv("Confidence: " + nf(results[0].confidence, 0, 2));
}
// draw() will not show anything until poses are found
function draw() {  //draw函数，每秒运行60次，不断地在画布上绘制图形
}
```

## 4、使用ml5的一些优缺点：
- 1、使用Web浏览器的内置的GPU进行计算，要比基于中央处理单元（CPU）的计算更快

- 2、正因为ml5的功能是基于浏览器的，只需在浏览器中运行，不会依赖专门的硬件、系统配置，而且是实时的。

- 3、目前的ml5设置不支持node.js，所有ml5.js功能都基于使用浏览器GPU

- 4、ml5现在提供的一些可用的模型都是基于英文的，没有基于中文的，基于中文的需要自己训练
