package de.htw.fp2.examples;

import org.apache.log4j.Logger;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

/**
 * XOR: This example is essentially the "Hello World" of neural network
 * programming.  This example shows how to construct an Encog neural
 * network to predict the output from the XOR operator.  This example
 * uses backpropagation to train the neural network.
 * <p/>
 * This example attempts to use a minimum of Encog features to create and
 * train the neural network.  This allows you to see exactly what is going
 * on.  For a more advanced example, that uses Encog factories, refer to
 * the XORFactory example.
 */
public class SimpleXOR {

    private static Logger log = Logger.getLogger(SimpleXOR.class.getName());

    /**
     * The input necessary for XOR.
     */
    public static double XOR_INPUT[][] = {{0.0, 0.0}, {1.0, 0.0},
            {0.0, 1.0}, {1.0, 1.0}};

    /**
     * The ideal data necessary for XOR.
     */
    public static double XOR_IDEAL[][] = {{0.0}, {1.0}, {1.0}, {0.0}};

    public boolean run() {

        try {
            // create a neural network, without using a factory
            BasicNetwork network = new BasicNetwork();
            network.addLayer(new BasicLayer(null, true, 2));
            network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
            network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
            network.getStructure().finalizeStructure();
            network.reset();

            // create training data
            MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

            // train the neural network
            final ResilientPropagation train = new ResilientPropagation(network, trainingSet);

            int epoch = 1;
            do {
                train.iteration();
                log.info("Epoch #" + epoch + " Error:" + train.getError());
                double[] weights = train.getCurrentFlatNetwork().getWeights();
                StringBuffer buffer = new StringBuffer();
                for(int i = 0; i <= weights.length-1; i++) {
                    buffer.append(String.valueOf(weights[i]) + ",");
                }
                // get weigths from Layer x neuron x to layer y neuron y
                //TODO
                //network.getWeight()

                log.info(buffer.toString());
                epoch++;
            } while (train.getError() > 0.01);
            train.finishTraining();

            // test the neural network
            log.info("Neural Network Results:");
            for (MLDataPair pair : trainingSet) {
                final MLData output = network.compute(pair.getInput());
                log.info(pair.getInput().getData(0) + "," + pair.getInput().getData(1)
                        + ", actual=" + output.getData(0) + ",ideal=" + pair.getIdeal().getData(0));
            }

            Encog.getInstance().shutdown();
        }catch(Exception ex){
            log.error(ex.getMessage());
            ex.printStackTrace();
            return false;
        }

        return true;
    }
}