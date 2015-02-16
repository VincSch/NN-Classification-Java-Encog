package de.htw.fp2.util;

import de.htw.fp2.common.Constants;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.Logger;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.Propagation;
import org.joda.time.DateTime;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by Vincent Schwarzer on 06.01.15.
 */
public class NetworkDebugUtility {

    private Logger log = Logger.getLogger(NetworkDebugUtility.class.getName());

    private BasicNetwork network;
    private Propagation trainer;
    private String fullDirectoryPath;
    private StringBuffer csv;
    private StringBuffer testResults;
    private StringBuffer weights;
    private boolean showWeights;
    private HtmlExportUtil htmlExportUtil;

    public NetworkDebugUtility(BasicNetwork network, Propagation trainer, boolean showWeights) {
        this.network = network;
        this.trainer = trainer;
        this.showWeights = showWeights;
        init();
    }

    private void init() {
        csv = new StringBuffer();
        testResults = new StringBuffer();
        weights = new StringBuffer();
        DateTime dt = new DateTime();
        DateTimeFormatter dtf = DateTimeFormat.forPattern(Constants.TIME_FORMAT);
        fullDirectoryPath = Constants.RESULTDIR + dt.toString(dtf);
        File directory = new File(fullDirectoryPath);
        directory.mkdir();
        String htmlReport = fullDirectoryPath + "/" + network.getClass().getSimpleName() + "_" + trainer.getClass().getSimpleName() + "_weights";
        htmlExportUtil = new HtmlExportUtil(htmlReport, network);
        initializeCSV();
    }

    public void iterationCallback(int epoch) {
        log.info("Epoch #" + epoch + " Error:" + trainer.getError());
        exportEpochErrorCSV(epoch);
        exportWeights(epoch);
    }

    public void generateTrainingReports() {
        saveCSV();
        htmlExportUtil.finalizeDocument();
    }

    public void generateTestReports() {
        saveTestResult();
    }

    private void exportWeights(int epoch) {
        htmlExportUtil.addTablesForIteration(epoch);
    }

    public void printTestResults(Map<MLDataPair, MLData> resultMap, String heading) {
        testResults.append("=======" + heading + "======= \n");
        for (Map.Entry<MLDataPair, MLData> entry : resultMap.entrySet()) {

            StringBuffer inputData = new StringBuffer(56);
            StringBuffer outputData = new StringBuffer(56);
            StringBuffer idealData = new StringBuffer(56);

            for (int i = 0; i < entry.getKey().getInput().getData().length; i++) {
                inputData.append(entry.getKey().getInput().getData(i) + " | ");
            }
            for (int i = 0; i < entry.getValue().getData().length; i++) {
                String outputStr = String.valueOf((Math.round(entry.getValue().getData(i) * 10) / 10.0));
                if (!outputStr.startsWith("-")) {
                    outputStr = " " + outputStr;
                }
                outputData.append(outputStr + " | ");
            }
            for (int i = 0; i < entry.getKey().getIdeal().getData().length; i++) {
                idealData.append(entry.getKey().getIdeal().getData(i) + " | ");
            }
            String result = ("input= " + inputData.toString()
                    + " actual= " + outputData.toString() + " ideal= " + idealData.toString());
            log.info(result.toString());

            testResults.append(result.toString() + "\n");
        }
    }

    private void exportEpochErrorCSV(int epoch) {
        csv.append('\n');
        csv.append(String.valueOf(epoch));
        csv.append(',');
        csv.append(String.valueOf(trainer.getError()));
        csv.append('\n');
    }

    private void initializeCSV() {
        csv.append("Iteration");
        csv.append(',');
        csv.append("Error");
    }

    private void saveCSV() {
        File file = new File(fullDirectoryPath + "/" + network.getClass().getSimpleName() + "_" + trainer.getClass().getSimpleName() + "_iteration_error.csv");
        try {
            FileUtils.writeStringToFile(file, csv.toString());
        } catch (IOException e) {
            log.error(e.getStackTrace());
        }
    }

    private void saveTestResult() {
        File file = new File(fullDirectoryPath + "/" + network.getClass().getSimpleName() + "_" + trainer.getClass().getSimpleName() + "_test_results.txt");
        try {
            FileUtils.writeStringToFile(file, testResults.toString());
        } catch (IOException e) {
            log.error(e.getStackTrace());
        }
    }
}
