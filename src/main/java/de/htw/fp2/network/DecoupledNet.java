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
        Basic_9_18_9, Basic_9_18_18_9, Basic_9_3_9,
        Linear_Decoupled_999, Linear_Pooling_Decoupled_999, ES_Decoupled_9_18_9,
        ES_Pooling_Decoupled_9_18_9,
        RandomDecoupled_9_18_9,
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
                case Basic_9_3_9:
                    createBasicNet_939();
                    break;
                case Basic_9_18_18_9:
                    createBasicNet_918189();
                    break;
                case Linear_Decoupled_999:
                    createLinearDecoupledNet_999();
                    break;
                case Linear_Pooling_Decoupled_999:
                    createLinearDecoupledNetWithPooling_999();
                    break;
                case ES_Decoupled_9_18_9:
                    createEverySecondDecoupledNet_9189();
                    break;
                case ES_Pooling_Decoupled_9_18_9:
                    createEverySecondDecoupledNetWithPooling_9189();
                    break;
                case RandomDecoupled_9_18_9:
                    createRandomDecoupledNet_9189();
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

    private void createBasicNet_939() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 3));
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

    private void createLinearDecoupledNet_999() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 9));
        basicLayers.add(new BasicLayer(null, false, 9));
        for (BasicLayer layer : basicLayers) {
            this.addLayer(layer);
        }
        this.getStructure().finalizeStructure();
        this.reset();

        //input to hidden
        for (int j = 0; j <= 8; j++) {
            for (int i = 0; i <= 8; i++) {
                if (i != j)
                    disableConnection(0, j, i);
            }
        }

        for (int j = 0; j <= 8; j++) {
            for (int i = 0; i <= 8; i++) {
                if (i != j)
                    disableConnection(1, j, i);
            }
        }
    }

    private void createLinearDecoupledNetWithPooling_999() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 9));
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(null, false, 9));
        for (BasicLayer layer : basicLayers) {
            this.addLayer(layer);
        }
        this.getStructure().finalizeStructure();
        this.reset();

        //input to hidden
        for (int j = 0; j <= 8; j++) {
            for (int i = 0; i <= 8; i++) {
                if (i != j)
                    disableConnection(0, j, i);
            }
        }

        for (int j = 0; j <= 8; j++) {
            for (int i = 0; i <= 8; i++) {
                if (i != j)
                    disableConnection(1, j, i);
            }
        }

    }

    private void createEverySecondDecoupledNet_9189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        for (BasicLayer layer : basicLayers) {
            this.addLayer(layer);
        }
        this.getStructure().finalizeStructure();
        this.reset();

        int y = 0;
        for (int j = 0; j <= 8; j++) {
            if (j != 0)
                y = y + 2;
            for (int i = 0; i <= 17; i++) {
                if (i != y)
                    disableConnection(0, j, i);
            }
        }


        for (int i = 0; i <= 8; i++) {
            if (i != 0)
                disableConnection(1, 0, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 1)
                disableConnection(1, 3, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 2)
                disableConnection(1, 5, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 3)
                disableConnection(1, 7, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 4)
                disableConnection(1, 9, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 5)
                disableConnection(1, 11, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 6)
                disableConnection(1, 13, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 7)
                disableConnection(1, 15, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 7)
                disableConnection(1, 17, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 1, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 2, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 4, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 6, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 8, i);
        }


        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 10, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 12, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 14, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 16, i);
        }
    }

    private void createEverySecondDecoupledNetWithPooling_9189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(null, false, 9));
        for (BasicLayer layer : basicLayers) {
            this.addLayer(layer);
        }
        this.getStructure().finalizeStructure();
        this.reset();

        int y = 0;
        for (int j = 0; j <= 8; j++) {
            if (j != 0)
                y = y + 2;
            for (int i = 0; i <= 17; i++) {
                if (i != y)
                    disableConnection(0, j, i);
            }
        }


        for (int i = 0; i <= 8; i++) {
            if (i != 0)
                disableConnection(1, 0, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 1)
                disableConnection(1, 3, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 2)
                disableConnection(1, 5, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 3)
                disableConnection(1, 7, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 4)
                disableConnection(1, 9, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 5)
                disableConnection(1, 11, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 6)
                disableConnection(1, 13, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 7)
                disableConnection(1, 15, i);
        }

        for (int i = 0; i <= 8; i++) {
            if (i != 7)
                disableConnection(1, 17, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 1, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 2, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 4, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 6, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 8, i);
        }


        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 10, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 12, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 14, i);
        }

        for (int i = 0; i <= 8; i++) {
            disableConnection(1, 16, i);
        }

    }

    private void createRandomDecoupledNet_9189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        for (BasicLayer layer : basicLayers) {
            this.addLayer(layer);
        }
        this.getStructure().finalizeStructure();
        this.reset();

        for (int j = 0; j <= 8; j++) {
            for (int i = 0; i <= 17; i++) {
                if (Math.random() > Math.random())
                    disableConnection(0, j, i);
            }
        }

        for (int j = 0; j <= 17; j++) {
            for (int i = 0; i <= 8; i++) {
                if (Math.random() < Math.random())
                    disableConnection(1, j, i);
            }
        }
    }

    private void testTopology() {

    }
}
