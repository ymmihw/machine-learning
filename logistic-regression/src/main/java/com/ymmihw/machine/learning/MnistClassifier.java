package com.ymmihw.machine.learning;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistClassifier {
  private static final Logger LOGGER = LoggerFactory.getLogger(MnistClassifier.class);
  private static final String BASE_PATH = System.getProperty("java.io.tmpdir") + "/mnist";

  public static void main(String[] args) throws Exception {
    int height = 28; // height of the picture in px
    int width = 28; // width of the picture in px
    int channels = 1; // single channel for grayscale images
    int outputNum = 10; // 10 digits classification
    int batchSize = 54; // number of samples that will be propagated through the network in each
                        // iteration
    int nEpochs = 1; // number of training epochs

    int seed = 1234; // number used to initialize a pseudorandom number generator.
    Random randNumGen = new Random(seed);

    LOGGER.info("Data load...");
    if (!new File(BASE_PATH + "/mnist_png").exists()) {
      File file = new File("mnist_png.tar.gz");
      Utils.extractTarArchive(file, BASE_PATH);
    }

    LOGGER.info("Data vectorization...");
    // vectorization of train data
    File trainData = new File(BASE_PATH + "/mnist_png/training");
    FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    // use parent directory name as the image label
    ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
    trainRR.initialize(trainSplit);
    DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

    // pixel values from 0-255 to 0-1 (min-max scaling)
    DataNormalization imageScaler = new ImagePreProcessingScaler();
    imageScaler.fit(trainIter);
    trainIter.setPreProcessor(imageScaler);

    // vectorization of test data
    File testData = new File(BASE_PATH + "/mnist_png/testing");
    FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
    testRR.initialize(testSplit);
    DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
    testIter.setPreProcessor(imageScaler); // same normalization for better results

    LOGGER.info("Network configuration and training...");
    // reduce the learning rate as the number of training epochs increases
    // iteration #, learning rate
    Map<Integer, Double> learningRateSchedule = new HashMap<>();
    learningRateSchedule.put(0, 0.06);
    learningRateSchedule.put(200, 0.05);
    learningRateSchedule.put(600, 0.028);
    learningRateSchedule.put(800, 0.0060);
    learningRateSchedule.put(1000, 0.001);

    ConvolutionLayer layer1 = new ConvolutionLayer.Builder(5, 5).nIn(channels).stride(1, 1).nOut(20)
        .activation(Activation.IDENTITY).build();
    SubsamplingLayer layer2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2).stride(2, 2).build();
    // nIn need not specified in later layers
    ConvolutionLayer layer3 = new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(50)
        .activation(Activation.IDENTITY).build();
    SubsamplingLayer layer4 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2).stride(2, 2).build();
    DenseLayer layer5 = new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build();
    OutputLayer layer6 = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum).activation(Activation.SOFTMAX).build();
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).l2(0.0005)
        .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
        .weightInit(WeightInit.XAVIER).list().layer(layer1).layer(layer2).layer(layer3)
        .layer(layer4).layer(layer5).layer(layer6)
        // InputType.convolutional for normal image
        .setInputType(InputType.convolutionalFlat(height, width, channels)).build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(10));
    LOGGER.info("Total num of params: {}", net.numParams());

    // evaluation while training (the score should go down)
    for (int i = 0; i < nEpochs; i++) {
      net.fit(trainIter);
      LOGGER.info("Completed epoch {}", i);
      Evaluation eval = net.evaluate(testIter);
      LOGGER.info(eval.stats());

      trainIter.reset();
      testIter.reset();
    }

    File ministModelPath = new File(BASE_PATH + "/minist-model.zip");
    ModelSerializer.writeModel(net, ministModelPath, true);
    LOGGER.info("The MINIST model has been saved in {}", ministModelPath.getPath());
  }
}
