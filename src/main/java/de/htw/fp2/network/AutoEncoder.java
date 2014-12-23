package de.htw.fp2.network;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

/**
 * Created by patrick on 13.12.14.
 */
public class AutoEncoder extends BasicNetwork {

    private BasicNetwork network;
    private int featureNeuron = 0;
    private int encodeNeuron = 0;

    public AutoEncoder(int input, int hidden) {
        super();
        this.featureNeuron = input;
        this.encodeNeuron = hidden;
        this.addLayer(new BasicLayer(null, true, input));
        this.addLayer(
            new BasicLayer(new ActivationSigmoid(), true, hidden));
        this.addLayer(new BasicLayer(new ActivationSigmoid(), false, input));
        this.getStructure().finalizeStructure();
        this.reset();
    }

}
