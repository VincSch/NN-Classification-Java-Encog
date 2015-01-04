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
        List<Pattern> idealOutputPattern = new ArrayList<>();
        try {
            patterns = PatternCreator
                    .readFrom(new File("src/main/resources/dataset/input.nn"));
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            idealOutputPattern = PatternCreator
                    .readFrom(new File("src/main/resources/dataset/idealOutput.nn"));
        } catch (Exception e) {
            e.printStackTrace();
        }


        String input = "";
        for (int i = 0; i < patterns.size(); i++) {
            BasicMLData inputData = new BasicMLData(
                    patterns.get(i).getFlat());

            BasicMLData outputData = new BasicMLData(
                    idealOutputPattern.get(i).getFlat());

            trainingSet.add(inputData, outputData);
        }

        DecoupledNet decoupledNet = createNet();
        decoupledNet.getStructure().finalizeStructure();
        decoupledNet.reset();

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
        for (MLDataPair pair : trainingSet) {
            final MLData output = decoupledNet.compute(pair.getInput());
            log.info(printResult(pair, output));
        }
        Encog.getInstance().shutdown();
        return true;
    }

    private DecoupledNet createNet() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        DecoupledNet decoupledNet = new DecoupledNet(basicLayers);
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
            outputData.append((Math.round(output.getData(i)*10)/10.0)+ " | ");
        }
        for (int i = 0; i < pair.getIdeal().getData().length; i++) {
            idealData.append(pair.getIdeal().getData(i) + " | ");
        }
        String result = ("input=" + inputData.toString()
                + " actual=" + outputData.toString() + " ideal=" + idealData.toString());
        return result;
    }
}
