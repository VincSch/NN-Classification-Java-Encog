package de.htw.fp2.examples;

import de.htw.fp2.train.DecoupledResilientPropagation;
import de.htw.fp2.util.NetworkPrinter;
import de.htw.fp2.dataset.Pattern;
import de.htw.fp2.dataset.PatternCreator;
import de.htw.fp2.network.DecoupledNet;
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
import java.io.FileWriter;
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

        DecoupledNet decoupledNet = createPartlyDecoupledNet_99189();
        NetworkPrinter prettyPrint = new NetworkPrinter(decoupledNet.getFlat());

        // train the neural network
        //final ResilientPropagation train = new ResilientPropagation(decoupledNet,
        //        trainingSet);

        final DecoupledResilientPropagation train = new DecoupledResilientPropagation(decoupledNet,
                trainingSet);

        int epoch = 1;
        try {
            FileWriter writer = new FileWriter("/Volumes/HDD/Data/Studium/Master/HTW-Berlin/3-Semester/Forschungsprojekt/fp2-neural-networks-encog/test99189DECResilent.csv");

            writer.append("Iteration");
            writer.append(',');
            writer.append("Error");
            writer.append('\n');

            do {
                train.iteration();
                writer.append(String.valueOf(epoch));
                writer.append(',');
                writer.append(String.valueOf(train.getError()));
                writer.append('\n');
                log.info("Epoch #" + epoch + " Error:" + train.getError());
                log.info(prettyPrint.printWeights());
                epoch++;
            } while (train.getError() > 0.01);
            train.finishTraining();
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
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

    private DecoupledNet createBasicNet_9189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        DecoupledNet decoupledNet = new DecoupledNet(basicLayers);
        return decoupledNet;
    }

    private DecoupledNet createPartlyDecoupledNet_9189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        DecoupledNet decoupledNet = new DecoupledNet(basicLayers);

        //input to hidden
        for(int i = 1; i >= 18; i++){
            if(i!=1)
                decoupledNet.disableConnection(0, 0, i);
        }
        for(int i = 0; i >= 17; i++){
            if(i!=3)
                decoupledNet.disableConnection(0, 1, i);
        }

        for(int i = 0; i >= 17; i++){
            if(i!=5)
                decoupledNet.disableConnection(0, 2, i);
        }

        for(int i = 0; i >= 17; i++){
            if(i!=7)
                decoupledNet.disableConnection(0, 3, i);
        }

        for(int i = 0; i >= 17; i++){
            if(i!=9)
                decoupledNet.disableConnection(0, 4, i);
        }

        for(int i = 0; i >= 11; i++){
            if(i!=9)
                decoupledNet.disableConnection(0, 5, i);
        }

        for(int i = 0; i >= 13; i++){
            if(i!=9)
                decoupledNet.disableConnection(0, 6, i);
        }

        for(int i = 0; i >= 15; i++){
            if(i!=9)
                decoupledNet.disableConnection(0, 7, i);
        }
        for(int i = 0; i >= 17; i++){
            if(i!=9)
                decoupledNet.disableConnection(0, 8, i);
        }



        return decoupledNet;
    }

    private DecoupledNet createPartlyDecoupledNet_99189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        DecoupledNet decoupledNet = new DecoupledNet(basicLayers);

        //input to hidden
        for(int i = 1; i >= 18; i++){
            if(i!=1)
                decoupledNet.disableConnection(1, 0, i);
        }
        for(int i = 0; i >= 17; i++){
            if(i!=3)
                decoupledNet.disableConnection(1, 1, i);
        }

        for(int i = 0; i >= 17; i++){
            if(i!=5)
                decoupledNet.disableConnection(1, 2, i);
        }

        for(int i = 0; i >= 17; i++){
            if(i!=7)
                decoupledNet.disableConnection(1, 3, i);
        }

        for(int i = 0; i >= 17; i++){
            if(i!=9)
                decoupledNet.disableConnection(1, 4, i);
        }

        for(int i = 0; i >= 11; i++){
            if(i!=9)
                decoupledNet.disableConnection(1, 5, i);
        }

        for(int i = 0; i >= 13; i++){
            if(i!=9)
                decoupledNet.disableConnection(1, 6, i);
        }

        for(int i = 0; i >= 15; i++){
            if(i!=9)
                decoupledNet.disableConnection(1, 7, i);
        }
        for(int i = 0; i >= 17; i++){
            if(i!=9)
                decoupledNet.disableConnection(1, 8, i);
        }



        return decoupledNet;
    }

}
