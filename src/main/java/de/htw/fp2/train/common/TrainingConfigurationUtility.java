package de.htw.fp2.train.common;

import de.htw.fp2.common.Constants;
import de.htw.fp2.dataset.Pattern;
import de.htw.fp2.dataset.PatternCreator;
import org.apache.log4j.Logger;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Vincent Schwarzer on 15.02.15.
 * This class eases the setup of our training data before starting to train a nn
 */
public class TrainingConfigurationUtility {

    private Logger log = Logger.getLogger(TrainingConfigurationUtility.class.getName());

    private MLDataSet trainingSet = new BasicMLDataSet();
    private List<Pattern> inputPattern = new ArrayList<>();
    private List<Pattern> idealOutputPattern = new ArrayList<>();
    private List<Pattern> noisyPattern = new ArrayList<>();


    /**
     * Constructor
     */
    public TrainingConfigurationUtility() {
        init();
    }

    /**
     * initialize training data
     */
    private void init() {
        try {
            loadPattern();
            createTrainingSet();
        } catch (Exception ex) {
            log.error("Something went wrong while initializing your training data!");
        }
    }

    private void loadPattern() throws Exception {
        inputPattern = PatternCreator
                .readFrom(new File(Constants.INPUT_PATTERN_PATH));
        idealOutputPattern = PatternCreator
                .readFrom(new File(Constants.OUTPUT_PATTERN_PATH));

        noisyPattern = PatternCreator
                .readFrom(new File(Constants.NOISY_PATTERN_PATH));
    }

    private void createTrainingSet() {
        for (int i = 0; i < inputPattern.size(); i++) {
            BasicMLData inputData = new BasicMLData(
                    inputPattern.get(i).getFlat());

            BasicMLData outputData = new BasicMLData(
                    idealOutputPattern.get(i).getFlat());

            trainingSet.add(inputData, outputData);
        }
    }

    public MLDataSet getTrainingSet() {
        return trainingSet;
    }

    public List<Pattern> getInputPattern() {
        return inputPattern;
    }

    public List<Pattern> getIdealOutputPattern() {
        return idealOutputPattern;
    }

    public List<Pattern> getNoisyPattern() {
        return noisyPattern;
    }

}
