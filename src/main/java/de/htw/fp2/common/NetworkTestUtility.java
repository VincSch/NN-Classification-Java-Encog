package de.htw.fp2.common;

import de.htw.fp2.train.common.TrainingConfigurationUtility;
import org.apache.log4j.Logger;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Vincent Schwarzer on 15.02.15.
 */
public class NetworkTestUtility {

    private Logger log = Logger.getLogger(NetworkTestUtility.class.getName());
    private TrainingConfigurationUtility trainingConfigurationUtility;
    private BasicNetwork network;

    /**
     * Constructor
     */
    public NetworkTestUtility(TrainingConfigurationUtility trainingConfigurationUtility, BasicNetwork network) {
        this.trainingConfigurationUtility = trainingConfigurationUtility;
        this.network = network;
    }

    public Map<MLDataPair, MLData> testNoisyData() {
        MLDataSet noisyTrainingSet = new BasicMLDataSet();
        for (int i = 0; i < trainingConfigurationUtility.getNoisyPattern().size(); i++) {
            BasicMLData inputData = new BasicMLData(
                    trainingConfigurationUtility.getNoisyPattern().get(i).getFlat());
            BasicMLData outputData = new BasicMLData(
                    trainingConfigurationUtility.getIdealOutputPattern().get(i).getFlat());
            noisyTrainingSet.add(inputData, outputData);
        }

        HashMap<MLDataPair, MLData> results = new HashMap<MLDataPair, MLData>();
        for (MLDataPair pair : noisyTrainingSet) {
            final MLData output = network.compute(pair.getInput());
            results.put(pair, output);
        }

        return results;
    }

    public Map<MLDataPair, MLData> testIdealData() {
        MLDataSet noisyTrainingSet = new BasicMLDataSet();
        for (int i = 0; i < trainingConfigurationUtility.getInputPattern().size(); i++) {
            BasicMLData inputData = new BasicMLData(
                    trainingConfigurationUtility.getInputPattern().get(i).getFlat());
            BasicMLData outputData = new BasicMLData(
                    trainingConfigurationUtility.getIdealOutputPattern().get(i).getFlat());
            noisyTrainingSet.add(inputData, outputData);
        }

        HashMap<MLDataPair, MLData> results = new HashMap<MLDataPair, MLData>();
        for (MLDataPair pair : noisyTrainingSet) {
            final MLData output = network.compute(pair.getInput());
            results.put(pair, output);
        }

        return results;
    }
}
