package de.htw.fp2.examples;

import de.htw.fp2.dataset.Pattern;
import de.htw.fp2.dataset.PatternCreator;
import de.htw.fp2.train.LayerWiseTrainer;
import de.htw.fp2.visualization.ImageCreator;
import org.apache.log4j.Logger;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by patrick on 15.12.14.
 */
public class StackedAutoEncoder {

    private static Logger log = Logger
        .getLogger(StackedAutoEncoder.class.getName());

    public static void main(String[] args) {
        // create training data
        List<Pattern> patterns = new ArrayList<>();
        try {
            patterns = PatternCreator
                .readFrom(new File("src/main/resources/dataset/input.nn"));
        } catch (Exception e) {
            e.printStackTrace();
        }
        double[][] inputData = new double[patterns.size()][9];
        double[][] idealData = new double[patterns.size()][3];
        try {
            String input = "";
            for (int i = 0; i < patterns.size(); i++) {
                double[] ideal = toBitArray(i);
                inputData[i] = patterns.get(i).getFlat();
                idealData[i] = ideal;
                input += patterns.get(i).toString() + "\n";
                double[][] inPattern = patterns.get(i).getDoubleValue();
                ImageCreator
                    .createGrayScale(new File("input" + i + ".bmp"), inPattern);
            }
        } catch (IOException e) {
            log.error(e);
        }
        BasicNetwork encoder = new BasicNetwork();
        encoder.addLayer(new BasicLayer(null, true, 9));
        encoder.addLayer(new BasicLayer(new ActivationSigmoid(), true, 32));
        encoder.addLayer(new BasicLayer(new ActivationSigmoid(), true, 12));
        encoder.addLayer(new BasicLayer(new ActivationSigmoid(), true, 6));
        encoder.addLayer(new BasicLayer(new ActivationSigmoid(), false, 3));
        encoder.getStructure().finalizeStructure(); //remove the layer structure end creates a flat structure
        encoder.reset();

        // train the neural network
        final LayerWiseTrainer train = new LayerWiseTrainer(encoder, inputData,
            idealData);
        train.train();
        train.fineTune();

        // test the neural network
        log.info("Neural Network Results:");
        for (int i = 0; i < inputData.length; i++) {
            final MLData output = encoder
                .compute(new BasicMLData(inputData[i]));
            log.info(Math.round(output.getData(0)) + ","
                + Math.round(output.getData(1)) + ","
                + Math.round(output.getData(2)) + ",");

        }
        Encog.getInstance().shutdown();
    }

    private static double[] toBitArray(int decimal) {
        double[] bitArray = new double[3];
        for (int i = bitArray.length - 1; i >= 0; i--) {
            double exp = Math.pow(2, i);
            if ((decimal - exp) >= 0) {
                bitArray[i] = 1.0;
                decimal -= exp;
            } else {
                bitArray[i] = 0.0;
            }
        }
        log.info(Arrays.toString(bitArray));
        return bitArray;
    }
}
