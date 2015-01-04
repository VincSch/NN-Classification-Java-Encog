package de.htw.fp2.examples;

import de.htw.fp2.common.Constants;
import de.htw.fp2.dataset.Pattern;
import de.htw.fp2.dataset.PatternCreator;
import de.htw.fp2.network.DecoupledNet;
import de.htw.fp2.visualization.ImageCreator;
import org.apache.log4j.Logger;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by vs on 04.01.15.
 */
public class DecoupledNetClassification {

    private Logger log = Logger.getLogger(DecoupledNetClassification.class.getName());

    public boolean run() {
        MLDataSet trainingSet = new BasicMLDataSet();
        List<Pattern> patterns = new ArrayList<>();
        List<Pattern> idealOutputPatterns = new ArrayList<>();
        List<Pattern> testNoisyPatterns = new ArrayList<>();
        try {
            patterns = PatternCreator
                    .readFrom(new File("src/main/resources/dataset/input.nn"));
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            idealOutputPatterns = PatternCreator
                    .readFrom(new File("src/main/resources/dataset/idealOutput.nn"));
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            testNoisyPatterns = PatternCreator
                    .readFrom(new File("src/main/resources/dataset/testData.nn"));
        } catch (Exception e) {
            e.printStackTrace();
        }



        String input = "";
        for (int i = 0; i < patterns.size(); i++) {
            BasicMLData inputData = new BasicMLData(
                    patterns.get(i).getFlat());

            BasicMLData outputData = new BasicMLData(
                    idealOutputPatterns.get(i).getFlat());

            trainingSet.add(inputData, outputData);
        }

        DecoupledNet decoupledNet = createNet();

        // train the neural network
        final ResilientPropagation train = new ResilientPropagation(decoupledNet,
                trainingSet);
        int epoch = 1;

        do {
            train.iteration();
            log.info("Epoch #" + epoch + " Error:" + train.getError());
            epoch++;
        } while (train.getError() > 0.01);
        train.finishTraining();

        // test the neural network
        log.info("Neural Network Results:");
        log.info("Learned examples:");
        for (MLDataPair pair : trainingSet) {
            final MLData output = decoupledNet.compute(pair.getInput());
            log.info(printResult(pair, output));
        }

        log.info("New examples with noise:");
        MLDataSet noisyTrainingSet = new BasicMLDataSet();
        for (int i = 0; i < testNoisyPatterns.size(); i++) {
            BasicMLData inputData = new BasicMLData(
                    testNoisyPatterns.get(i).getFlat());
            BasicMLData outputData = new BasicMLData(
                    idealOutputPatterns.get(i).getFlat());
            noisyTrainingSet.add(inputData, outputData);
        }
        for (MLDataPair pair : noisyTrainingSet) {
            final MLData output = decoupledNet.compute(pair.getInput());
            log.info(printResult(pair, output));
        }
        Encog.getInstance().shutdown();
        return true;
    }

    private DecoupledNet createNet() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 9));
        basicLayers.add(new BasicLayer(null, false, 9));
        DecoupledNet decoupledNet = new DecoupledNet(basicLayers);
        //decouple(decoupledNet);
        return decoupledNet;
    }

    private String printResult(MLDataPair pair, MLData output) {
        StringBuffer inputData = new StringBuffer(56);
        StringBuffer outputData = new StringBuffer(56);
        StringBuffer idealData = new StringBuffer(56);

        for (int i = 0; i < pair.getInput().getData().length; i++) {
            inputData.append(pair.getInput().getData(i) + " | ");
        }
        for (int i = 0; i < output.getData().length; i++) {
            String outputStr = String.valueOf((Math.round(output.getData(i) * 10) / 10.0));
            if (!outputStr.startsWith("-")) {
                outputStr = " " + outputStr;
            }
            outputData.append(outputStr + " | ");
        }
        for (int i = 0; i < pair.getIdeal().getData().length; i++) {
            idealData.append(pair.getIdeal().getData(i) + " | ");
        }
        String result = ("input= " + inputData.toString()
                + " actual= " + outputData.toString() + " ideal= " + idealData.toString());
        return result;
    }

    private void decouple(DecoupledNet decoupledNet){
        //0
        decoupledNet.disableConnection(0, 0, 1);
        decoupledNet.disableConnection(0, 0, 2);
        decoupledNet.disableConnection(0, 0, 3);
        decoupledNet.disableConnection(0, 0, 4);
        decoupledNet.disableConnection(0, 0, 5);
        decoupledNet.disableConnection(0, 0, 6);
        decoupledNet.disableConnection(0, 0, 7);
        decoupledNet.disableConnection(0, 0, 8);
        //1
        decoupledNet.disableConnection(0, 1, 0);
        decoupledNet.disableConnection(0, 1, 2);
        decoupledNet.disableConnection(0, 1, 3);
        decoupledNet.disableConnection(0, 1, 4);
        decoupledNet.disableConnection(0, 1, 5);
        decoupledNet.disableConnection(0, 1, 6);
        decoupledNet.disableConnection(0, 1, 7);
        decoupledNet.disableConnection(0, 1, 8);
        //2
        decoupledNet.disableConnection(0, 2, 0);
        decoupledNet.disableConnection(0, 2, 1);
        decoupledNet.disableConnection(0, 2, 3);
        decoupledNet.disableConnection(0, 2, 4);
        decoupledNet.disableConnection(0, 2, 5);
        decoupledNet.disableConnection(0, 2, 6);
        decoupledNet.disableConnection(0, 2, 7);
        decoupledNet.disableConnection(0, 2, 8);
        //3
        decoupledNet.disableConnection(0, 3, 0);
        decoupledNet.disableConnection(0, 3, 1);
        decoupledNet.disableConnection(0, 3, 2);
        decoupledNet.disableConnection(0, 3, 4);
        decoupledNet.disableConnection(0, 3, 5);
        decoupledNet.disableConnection(0, 3, 6);
        decoupledNet.disableConnection(0, 3, 7);
        decoupledNet.disableConnection(0, 3, 8);
        //4
        decoupledNet.disableConnection(0, 4, 0);
        decoupledNet.disableConnection(0, 4, 1);
        decoupledNet.disableConnection(0, 4, 2);
        decoupledNet.disableConnection(0, 4, 3);
        decoupledNet.disableConnection(0, 4, 5);
        decoupledNet.disableConnection(0, 4, 6);
        decoupledNet.disableConnection(0, 4, 7);
        decoupledNet.disableConnection(0, 4, 8);
        //5
        decoupledNet.disableConnection(0, 5, 0);
        decoupledNet.disableConnection(0, 5, 1);
        decoupledNet.disableConnection(0, 5, 2);
        decoupledNet.disableConnection(0, 5, 3);
        decoupledNet.disableConnection(0, 5, 4);
        decoupledNet.disableConnection(0, 5, 6);
        decoupledNet.disableConnection(0, 5, 7);
        decoupledNet.disableConnection(0, 5, 8);
        //6
        decoupledNet.disableConnection(0, 6, 0);
        decoupledNet.disableConnection(0, 6, 1);
        decoupledNet.disableConnection(0, 6, 2);
        decoupledNet.disableConnection(0, 6, 3);
        decoupledNet.disableConnection(0, 6, 4);
        decoupledNet.disableConnection(0, 6, 5);
        decoupledNet.disableConnection(0, 6, 7);
        decoupledNet.disableConnection(0, 6, 8);
        //7
        decoupledNet.disableConnection(0, 7, 0);
        decoupledNet.disableConnection(0, 7, 1);
        decoupledNet.disableConnection(0, 7, 2);
        decoupledNet.disableConnection(0, 7, 3);
        decoupledNet.disableConnection(0, 7, 4);
        decoupledNet.disableConnection(0, 7, 5);
        decoupledNet.disableConnection(0, 7, 6);
        decoupledNet.disableConnection(0, 7, 8);
        //8
        decoupledNet.disableConnection(0, 8, 0);
        decoupledNet.disableConnection(0, 8, 1);
        decoupledNet.disableConnection(0, 8, 2);
        decoupledNet.disableConnection(0, 8, 3);
        decoupledNet.disableConnection(0, 8, 4);
        decoupledNet.disableConnection(0, 8, 5);
        decoupledNet.disableConnection(0, 8, 6);
        decoupledNet.disableConnection(0, 8, 7);
    }
}
