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

    private List<Weight> ignoreWeights = new ArrayList<>();
    private final double NULL_VALUE = 0.0;
    private Topology topology;

    public enum Topology {
        Basic_9_18_9, Basic_9_18_18_9, Basic_9_64_9,
        Decoupled_9_18_9, Decoupled_9_9_18_9, Decoupled_9_64_9,
        RandomDecoupled_9_18_9, RandomDecoupled_9_18_18_9, RandomDecoupled_9_64_9,
        Not_Set, Test
    }

    public DecoupledNet(Topology topology) {
        super();
        this.topology = topology;
        if (topology == null)
            createBasicNet_9189();
        else {
            switch (topology) {
                case Basic_9_18_9:
                    createBasicNet_9189();
                    break;
                case Basic_9_18_18_9:
                    createBasicNet_918189();
                    break;
                case Basic_9_64_9:
                    createBasicNet_9189();
                    break;
                case Decoupled_9_18_9:
                    createPartlyDecoupledNet_9189();
                    break;
                case Decoupled_9_9_18_9:
                    createPartlyDecoupledNet_99189();
                    break;
                case Test:
                    testTopology();
                    break;
                default:
                    createBasicNet_9189();
                    break;
            }
        }
    }

    public DecoupledNet(List<BasicLayer> layerList, Topology topology) {
        super();
        this.topology = topology;
        for (BasicLayer layer : layerList) {
            this.addLayer(layer);
        }
        this.reset();
    }

    public void resetDisabledConnections() {
        for (Weight weight : ignoreWeights) {
            setWeight(weight.getFromLayer(), weight.getFromNeuron(), weight.getToNeuron(), NULL_VALUE);
        }
    }

    public void disableConnection(int fromLayer, int fromNeuron, int toNeuron) {
        setWeight(fromLayer, fromNeuron, toNeuron, NULL_VALUE);
        ignoreWeights.add(new Weight(fromLayer, fromNeuron, toNeuron));

    }

    public class Weight {

        private int fromLayer;
        private int fromNeuron;
        private int toNeuron;

        public Weight(int fromLayer, int fromNeuron, int toNeuron) {
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

    public Topology getTopology() {
        return topology;
    }

    private void createBasicNet_9189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        for (BasicLayer layer : basicLayers) {
            this.addLayer(layer);
        }
        this.getStructure().finalizeStructure();
        this.reset();
    }

    private void createBasicNet_918189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        for (BasicLayer layer : basicLayers) {
            this.addLayer(layer);
        }
        this.getStructure().finalizeStructure();
        this.reset();
    }

    private void createPartlyDecoupledNet_9189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        for (BasicLayer layer : basicLayers) {
            this.addLayer(layer);
        }
        this.getStructure().finalizeStructure();
        this.reset();

        //input to hidden
        for (int i = 1; i <= 17; i++) {
            if (i != 1)
                disableConnection(0, 0, i);
        }

        for (int i = 0; i <= 17; i++) {
            if (i != 3)
                disableConnection(0, 1, i);
        }

        for (int i = 0; i <= 17; i++) {
            if (i != 5)
                disableConnection(0, 2, i);
        }

        for (int i = 0; i <= 17; i++) {
            if (i != 7)
                disableConnection(0, 3, i);
        }

        for (int i = 0; i <= 17; i++) {
            if (i != 9)
                disableConnection(0, 4, i);
        }

        for (int i = 0; i <= 11; i++) {
            if (i != 9)
                disableConnection(0, 5, i);
        }

        for (int i = 0; i <= 13; i++) {
            if (i != 9)
                disableConnection(0, 6, i);
        }

        for (int i = 0; i <= 15; i++) {
            if (i != 9)
                disableConnection(0, 7, i);
        }
        for (int i = 0; i <= 17; i++) {
            if (i != 9)
                disableConnection(0, 8, i);
        }
    }

    private void createPartlyDecoupledNet_99189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        for (BasicLayer layer : basicLayers) {
            this.addLayer(layer);
        }
        this.getStructure().finalizeStructure();
        this.reset();

        //input to hidden
        for (int i = 1; i <= 18; i++) {
            if (i != 1)
                disableConnection(1, 0, i);
        }
        for (int i = 0; i <= 17; i++) {
            if (i != 3)
                disableConnection(1, 1, i);
        }

        for (int i = 0; i <= 17; i++) {
            if (i != 5)
                disableConnection(1, 2, i);
        }

        for (int i = 0; i <= 17; i++) {
            if (i != 7)
                disableConnection(1, 3, i);
        }

        for (int i = 0; i <= 17; i++) {
            if (i != 9)
                disableConnection(1, 4, i);
        }

        for (int i = 0; i <= 11; i++) {
            if (i != 9)
                disableConnection(1, 5, i);
        }

        for (int i = 0; i <= 13; i++) {
            if (i != 9)
                disableConnection(1, 6, i);
        }

        for (int i = 0; i <= 15; i++) {
            if (i != 9)
                disableConnection(1, 7, i);
        }
        for (int i = 0; i <= 17; i++) {
            if (i != 9)
                disableConnection(1, 8, i);
        }
    }

    private void testTopology() {

    }
}
