# Core-ML
### Core ML 概述
借助 Core ML，我们可以将经过训练的机器学习模型整合到我们的应用程序中。
![](http://upload-images.jianshu.io/upload_images/6365912-84a5d1e3a0e74a5a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

一个训练有素的模型是将机器学习算法来训练数据集合的结果。该模型根据新的输入数据进行预测。例如，一个在某个地区的历史房价上受过培训的模型可能能够在给定卧室和浴室的数量时预测房屋的价格。

Core ML 是领域特定框架和功能的基础。Core ML 支持[[Vision](https://developer.apple.com/documentation/vision?language=objc)
](https://developer.apple.com/documentation/vision?language=objc)图像分析，[[Natural Language](https://developer.apple.com/documentation/naturallanguage?language=objc)
](https://developer.apple.com/documentation/naturallanguage?language=objc)的自然语言处理，以及评估学习决策树的[GameplayKit](https://developer.apple.com/documentation/gameplaykit?language=objc)。Core ML 本身建立在像[Accelerate](https://developer.apple.com/documentation/accelerate?language=objc)和[BNNS](https://developer.apple.com/documentation/accelerate/bnns?language=objc)这样的低级原语以及[[Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders?language=objc)
](https://developer.apple.com/documentation/metalperformanceshaders?language=objc)[](https://developer.apple.com/documentation/accelerate/bnns?language=objc)之上。
![](http://upload-images.jianshu.io/upload_images/6365912-094e8ef7782e9847.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Core ML 针对器件性能进行了优化，最大限度地减少了内存占用量和功耗。在设备上严格运行可确保用户数据的隐私，并确保我们的应用在网络连接不可用时保持功能和响应。

### 获取 Core ML 模型
Core ML 支持多种机器学习模型，包括神经网络，tree ensembles，支持向量机和广义线性模型。Core ML 需要 Core ML 模型格式（带有`.mlmodel`文件扩展名的模型）。

使用 [Create ML](https://developer.apple.com/documentation/create_ml?language=objc) 和我们的数据，我们可以训练自定义模型来执行任务，如识别图像，从文本中提取含义或查找数值之间的关系。使用 Create ML 进行培训的模型采用 Core ML 模型格式，可以在我们的应用程序中使用。

Apple还提供了几种已经是 Core ML 模型格式的流行的开源[模型](https://developer.apple.com/machine-learning)。我们可以下载这些模型并在我们的应用中使用它们。另外，各种研究团队和大学都会发布他们的模型和培训数据，这些数据可能不在 Core ML 模型中。如果我们要在应用中使用这些模型，需要转换它们，如[受训模型转换为 Core ML](https://developer.apple.com/documentation/coreml/converting_trained_models_to_core_ml?language=objc)。

### 集成 Core ML 模型
##### 将模型添加到 Xcode 项目中
这里我使用的是已经训练好的[MoblieNet](https://developer.apple.com/machine-learning/run-a-model/)模型，将模型拖放到项目导航栏中，即可将模型添加到项目中：
![](https://upload-images.jianshu.io/upload_images/6365912-dc4777c5846f4ad3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在Compile Source中添加 `.mlmodel` 文件：
![Compile Source](https://upload-images.jianshu.io/upload_images/6365912-3b77c8794d4f1ca5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在 Xcode 中查看模型的信息：
![model](https://upload-images.jianshu.io/upload_images/6365912-c07e16760c8aba07.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
我们可以看到具体的模型类 MoblieNet，数据输入格式以及输出格式，在这里输入的是 image 格式的图片，输出的是字典和字符串。
##### 用代码创建模型
使用模型类的初始化程序来创建模型：MobileNet
```
let model = MobileNet()
```

##### 获取输入值传递给模型
在这里我们将相册中选中的图片传给模型：
```
let image = info[UIImagePickerControllerOriginalImage] as! UIImage
        
let pixelBuffer = image.pixelBuffer(width: 224, height: 224)
```
##### 使用模型进行预测
在这里我们将模型返回的值显示到界面中，包括结果以及匹配度：
```
let output = try?model.prediction(image: pixelBuffer!)
        
let probs = output?.classLabelProbs.sorted { $0.value > $1.value }
        
if let prob = probs?.first {
    resultLabel.text = "结果：\(prob.key) "

    let probText = String(format:"%.2f %", prob.value * 100)
            
    probLabel.text = "匹配度：\(probText)%"
}
```
![demo](https://upload-images.jianshu.io/upload_images/6365912-2c95eaf7fa8fd37a.gif?imageMogr2/auto-orient/strip)

[demo 下载](https://github.com/hzsss/Core-ML)

### 使用 Create ML 创建模型
在 WWDC18 中，苹果发布了最新的 Xcode10，在 Xcode10 中使用 Create ML 可以自己创建机器学习模型。

图像分类器是经过训练识别图像的机器学习模型。当我们给它一张图片时，它会返回图片的名称。
![](http://upload-images.jianshu.io/upload_images/6365912-a816249c3669fe6a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
通过显示大量已标记图像的示例来训练图像分类器。例如，我们可以通过展示大象，长颈鹿，狮子等各种照片来训练图像分类器来识别野生动物园动物。
![](http://upload-images.jianshu.io/upload_images/6365912-7bc90d02d00b6dde.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 准备数据
首先准备好用来训练和评估分类器的数据。选择约80％的图像创建一个训练数据集。从剩余的图像创建一个测试数据集。确保任何给定的图像仅出现在这两组中的一组中。

接下来，创建一个名为 Training Data 的文件夹，另一个名为 Testing Data。在每个文件夹中，使用我们的标签作为名称创建子文件夹。然后将图像分类到每个数据集合适的子文件夹中。
![](http://upload-images.jianshu.io/upload_images/6365912-9a5ae81b087236e7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
我们不需要以任何特定方式命名图像文件或向其添加元数据。只需将它们放入具有正确标签的文件夹中即可。

每个标签至少使用10个图像作为训练集，当然了，越多越好。此外，需要均衡每个种类的图像数量。例如，不要使用10张猎豹的图像和1000大象张图像。

图像可以是统一类型标识符合`public.image`的任何格式。这包括常见的格式，如JPEG和PNG。这些图像不需要是相同的尺寸，也不需要任何特定的尺寸，但最好使用至少为299x299像素的图像。如果可能的话，训练收集的图像的方式与收集图像以进行预测的方式类似。

需要提供多种图像。例如，使用从不同角度和不同照明条件下显示动物的图像。对于给定标签几乎相同的图像进行训练的分类器往往比在不同图像集上训练的分类器性能更差。

##### 在 Playground 显示图像分类器
准备好数据后，用 macOS 中创建一个新的 Xcode Playground。使用 Playground 创建 MLImageClassifierBuilder 实例并在实时视图中显示它：
```
// Import CreateMLUI to train the image classifier in the UI.
// For other Create ML tasks, import CreateML instead.
import CreateMLUI 

let builder = MLImageClassifierBuilder()
builder.showInLiveView()
```
![](http://upload-images.jianshu.io/upload_images/6365912-1bf4208b1d96bfbe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 训练模型
将 Training Data 文件夹从 Finder 拖到实时视图中的指定位置。训练过程开始，图像分类器显示其进展：
![](http://upload-images.jianshu.io/upload_images/6365912-629cc26f9abc559f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
作为训练过程的一部分，图像分类器会自动将训练数据分成训练集和验证集。这些都会影响训练，但方式各不相同。由于分割是随机完成的，因此每次训练模型时可能会得到不同的结果。

训练结束后，实时视图会显示培训和验证的准确性。这些报告了训练好的模型如何将来自相应组的图像分类。因为使用了这些图像训练模型，所以可以对它们进行很好的分类。
![](http://upload-images.jianshu.io/upload_images/6365912-c9abef1215c4eb55.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
##### 评估分类器的性能
接下来，通过使用以前从未使用过的图像进行测试，评估训练好的模型的性能。使用在开始训练之前创建的测试数据集。就像使用培训数据一样，将 Test Data 文件夹拖到实时视图中。
![](http://upload-images.jianshu.io/upload_images/6365912-52e47d4eb409b653.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

该模型处理所有的图像，为每个图像做出预测。因为这是标记数据，所以模型可以检查自己的预测。然后它将整体评估准确性作为 UI 中的最终指标。
![](http://upload-images.jianshu.io/upload_images/6365912-8bc4dc08d906d5e2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
如果评估表现不够好，可以重新培训更多数据，或更改其他一些培训配置。有关如何进行更详细的模型评估以及改进模型性能的策略的信息，可以参阅[提高模型的准确性](https://developer.apple.com/documentation/create_ml/improving_your_model_s_accuracy)。

##### 保存 Core ML 模型
当模型运行良好时，保存它以便在我们的应用程序中使用它。

给分类器一个有意义的名字。通过将 UI 中默认的 ImageClassifier 改为 AnimalClassifier。还可以添加更多关于模型的信息，例如作者和简介。点击显示三角形显示这些元数据字段并填写详细信息。
![](http://upload-images.jianshu.io/upload_images/6365912-3c40275f0a71c87b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
点击保存。将该模型保存到指定目录中。





