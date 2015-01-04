package de.htw.fp2.examples;

import de.htw.fp2.common.Constants;
import de.htw.fp2.dataset.Pattern;
import de.htw.fp2.dataset.PatternCreator;
import de.htw.fp2.util.ArrayUtil;
import de.htw.fp2.visualization.ImageCreator;
import org.apache.log4j.Logger;
import org.encog.Encog;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by patrick on 15.12.14.
 */
public class AutoEncoder {

    private Logger log = Logger.getLogger(AutoEncoder.class.getName());

    public boolean run() {
        // create training data
        MLDataSet trainingSet = new BasicMLDataSet();
        List<Pattern> patterns = new ArrayList<>();
        try {
            patterns = PatternCreator
                    .readFrom(new File("src/main/resources/dataset/input.nn"));
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            String input = "";
            for (int i = 0; i < patterns.size(); i++) {
                BasicMLData inputData = new BasicMLData(
                        patterns.get(i).getFlat());
                trainingSet.add(inputData, inputData);
                input += patterns.get(i).toString() + "\n";
                double[][] inPattern = patterns.get(i).getDoubleValue();
                ImageCreator
                        .createGrayScale(new File(Constants.INPUTDIR + "input" + i + ".bmp"), inPattern);
            }
        } catch (IOException e) {
            log.error(e);
        }
        de.htw.fp2.network.AutoEncoder encoder =
                new de.htw.fp2.network.AutoEncoder(9, 32);

        // train the neural network
        final ResilientPropagation train = new ResilientPropagation(encoder,
                trainingSet);
        int epoch = 1;

        do {
            train.iteration();
            //            layers = encoder.getStructure().getLayers();
            encoder.getWeight(0, 0, 0);
            //            log.info("Epoch #" + epoch + " Error:" + train.getError());
            //            double[] weights = train.getCurrentFlatNetwork().getWeights();
            //            StringBuffer buffer = new StringBuffer();
            //            for (int i = 0; i <= weights.length - 1; i++) {
            //                buffer.append(String.valueOf(weights[i]) + ",");
            //            }
            //            log.info(buffer.toString());
            epoch++;
        } while (train.getError() > 0.01);
        train.finishTraining();

        // test the neural network
        log.info("Neural Network Results:");

        for (int i = 0; i < trainingSet.size(); i++) {
            MLDataPair pair = trainingSet.get(i);
            final MLData output = encoder.compute(pair.getInput());
            int[] out = new int[output.size()];
            StringBuilder sb = new StringBuilder("[");
            int counter = 0;
            for (double value : output.getData()) {
                out[counter++] = (int) Math.round(value);
                sb.append(Math.round(value));
                sb.append(",");
            }
            sb.append("]");
            log.info(sb);
            int[][] outPattern = ArrayUtil.toMatrix(out, 3);
            try {
                ImageCreator
                        .createGrayScale(new File(Constants.OUTPUTDIR + "output" + i + ".bmp"),
                                outPattern);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        Encog.getInstance().shutdown();
        return true;
    }
}
