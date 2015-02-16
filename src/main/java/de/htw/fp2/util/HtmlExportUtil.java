package de.htw.fp2.util;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Logger;
import org.encog.neural.networks.BasicNetwork;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by vs on 16.02.15.
 */
public class HtmlExportUtil {
    private Logger log = Logger.getLogger(HtmlExportUtil.class.getName());
    private BasicNetwork network;
    private StringBuffer document;
    String documentPath;

    public HtmlExportUtil(String documentPath, BasicNetwork network) {
        document = new StringBuffer();
        this.network = network;
        this.documentPath = documentPath;
        createDocument();
    }

    private void head() {
        StringBuffer head = new StringBuffer();
        head.append(
                "<!DOCTYPE html>" +
                        "<html>" +
                        "<head lang=\"en\">" +
                        "<link href=\"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css\" rel=\"stylesheet\">" +
                        "<script src=\"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/js/bootstrap.min.js\"></script>" +
                        "    <meta charset=\"UTF-8\">" +
                        "    <title>" + documentPath + "</title>" +
                        "</head>" +
                        "<body>" +
                        "<div align=\"center\">"
        );
        document.append(head.toString());
    }

    private void createTable(int headerCount) {
        document.append("<table class=\"table table-bordered\" style=\"width:50%\">");
        document.append("<tr>");
        document.append("<th>#</th>");
        for (int i = headerCount; i > 0; i--) {
            document.append("<th> IN " + i + "</th>");
        }
        document.append("</tr>");
    }

    private void closeTable() {
        document.append("</table> <br>");
    }

    private void addRow(List<String> scalar, int counter) {
        StringBuffer row = new StringBuffer();
        row.append("<tr>");
        row.append("<td> <strong> OUT " + counter + "</strong></td>");
        for (String value : scalar) {
            row.append("<td>" + value + "</td>");
        }
        row.append("<tr>");
        document.append(row.toString());
    }

    private void tale() {
        document.append("</div></body></html>");
    }

    private void h2(int epoch) {
        document.append("<h2> Iteration:" + epoch + "</h2> <hr>");
    }

    public void createDocument() {
        head();
    }

    public void finalizeDocument() {
        tale();
        File file = new File(documentPath + ".html");
        try {
            FileUtils.writeStringToFile(file, document.toString());
        } catch (IOException e) {
            log.error(e.getStackTrace());
        }
    }

    public void addTablesForIteration(int epoch) {
        h2(epoch);
        double[] weights = network.getFlat().getWeights();
        int[] neuronsPerLayer = network.getFlat().getLayerCounts();

        int iteration = neuronsPerLayer.length;
        StringBuffer weightMatrixes = new StringBuffer();

        for (int i = iteration - 1; i > 0; i--) {
            //weightMatrixes.append("\nLayer: " + (i + 1));
            document.append("<h3>Layer " + Math.abs(new Integer(i - iteration)) + " zu Layer " + (Math.abs(new Integer(i - iteration)) + 1) + "</h4>");
            createTable(neuronsPerLayer[i]);
            int countOfWeights = neuronsPerLayer[i] * neuronsPerLayer[i - 1];
            int countInputNeurons = neuronsPerLayer[i];
            int stopAt = weights.length - (countOfWeights + 1);

            List<String> weightScalar = new ArrayList<String>();
            int counter;
            if (i > 0) {
                counter = neuronsPerLayer[i - 1];
            } else {
                counter = neuronsPerLayer[i];
            }
            for (int j = weights.length - 1; j >= stopAt; j--) {
                countInputNeurons--;
                weightScalar.add(String.valueOf((Math.round(weights[j] * 10) / 10.0)));
                //weightMatrixes.append("\n" + String.valueOf((Math.round(weights[j] * 10) / 10.0)));
                if (countInputNeurons == 0) {
                    addRow(weightScalar, counter);
                    counter--;
                    countInputNeurons = neuronsPerLayer[i];
                    weightScalar.clear();
                    //  weightMatrixes.append("\n ==============");
                }
            }
            closeTable();
        }
    }
}
