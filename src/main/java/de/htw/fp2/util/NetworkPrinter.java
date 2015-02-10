package de.htw.fp2.util;

import org.apache.log4j.Logger;
import org.encog.neural.flat.FlatNetwork;

/**
 * Created by vs on 06.01.15.
 */
public class NetworkPrinter {

    private Logger log = Logger.getLogger(NetworkPrinter.class.getName());

    private FlatNetwork flatNetwork;
    private StringBuffer buffer;

    public NetworkPrinter(FlatNetwork flatnetwork) {
        this.flatNetwork = flatnetwork;
    }


    public String printWeights() {
        double[] weights = flatNetwork.getWeights();
        int[] neuronsPerLayer = flatNetwork.getLayerCounts();

        int iteration = neuronsPerLayer.length;
        StringBuffer weightMatrixes = new StringBuffer();

        for (int i = iteration - 1; i > 0; i--) {
            weightMatrixes.append("\nLayer: " + (i+1));
            int countOfWeights = neuronsPerLayer[i] * neuronsPerLayer[i - 1];
            int countInputNeurons = neuronsPerLayer[i];
            int stopAt = weights.length - (countOfWeights + 1);
            for (int j = weights.length - 1; j >= stopAt; j--) {
                countInputNeurons--;
                weightMatrixes.append("\n" + String.valueOf((Math.round(weights[j] * 10) / 10.0)));
                if(countInputNeurons == 0){
                    countInputNeurons = neuronsPerLayer[i];
                    weightMatrixes.append("\n ==============");
                }
            }
        }
        //System.lineSeparator()
        return weightMatrixes.toString();
    }

}
