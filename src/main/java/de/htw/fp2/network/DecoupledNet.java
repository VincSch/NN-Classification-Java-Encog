package de.htw.fp2.network;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vs on 03.01.15.
 */
public class DecoupledNet extends BasicNetwork {

    public List<Weight> ignoreWeights = new ArrayList<>();

    public DecoupledNet(List<BasicLayer> layerList){
        super();
        for(BasicLayer layer : layerList){
            this.addLayer(layer);
        }
    }

    public void disableConnection(int fromLayer, int fromNeuron, int toNeuron){
        enableConnection(fromLayer, fromNeuron, toNeuron, false);
        this.ignoreWeights.add(new Weight(fromLayer, fromNeuron, toNeuron));
    }

    public class Weight {

        private int fromLayer;
        private int fromNeuron;
        private int toNeuron;

        public Weight(int fromLayer, int fromNeuron, int toNeuron){
            this.fromLayer = fromLayer;
            this.fromNeuron = fromNeuron;
            this.toNeuron = toNeuron;
        }

        public int getFromLayer() {
            return fromLayer;
        }

        public void setFromLayer(int fromLayer) {
            this.fromLayer = fromLayer;
        }

        public int getFromNeuron() {
            return fromNeuron;
        }

        public void setFromNeuron(int fromNeuron) {
            this.fromNeuron = fromNeuron;
        }

        public int getToNeuron() {
            return toNeuron;
        }

        public void setToNeuron(int toNeuron) {
            this.toNeuron = toNeuron;
        }
    }

    public List<Weight> getIgnoreWeights() {
        return ignoreWeights;
    }

    public void setIgnoreWeights(List<Weight> ignoreWeights) {
        this.ignoreWeights = ignoreWeights;
    }
}
