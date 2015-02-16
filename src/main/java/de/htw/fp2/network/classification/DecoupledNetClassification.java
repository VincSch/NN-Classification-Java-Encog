package de.htw.fp2.network.classification;

import de.htw.fp2.common.NetworkTestUtility;
import de.htw.fp2.train.DecoupledResilientPropagation;
import de.htw.fp2.train.common.TrainingConfigurationUtility;
import de.htw.fp2.util.NetworkDebugUtility;
import de.htw.fp2.network.DecoupledNet;
import org.apache.log4j.Logger;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Vincent Schwarzer on 04.01.15.
 */
public class DecoupledNetClassification {

    private Logger log = Logger.getLogger(DecoupledNetClassification.class.getName());
    private boolean useDecoupledResilient;
    private TrainingConfigurationUtility trainingConfig;
    private DecoupledNet decoupledNet;
    private Propagation trainer;
    private NetworkDebugUtility networkDebugger;
    private NetworkTestUtility testUtility;
    private double error;

    public DecoupledNetClassification(boolean useDecoupledResilient) {
        this.useDecoupledResilient = useDecoupledResilient;
        this.error = 0.01;
        trainingConfig = new TrainingConfigurationUtility();
        this.decoupledNet = createBasicNet_9189();

        if (this.useDecoupledResilient)
            this.trainer = new DecoupledResilientPropagation(decoupledNet,
                    trainingConfig.getTrainingSet());
        else
            this.trainer = new ResilientPropagation(decoupledNet,
                    trainingConfig.getTrainingSet());

        networkDebugger = new NetworkDebugUtility(this.decoupledNet, this.trainer, false);
        testUtility = new NetworkTestUtility(trainingConfig, decoupledNet);
    }

    public DecoupledNetClassification(boolean useDecoupledResilient, double error, boolean showWeights) {
        this.useDecoupledResilient = useDecoupledResilient;
        this.error = error;
        trainingConfig = new TrainingConfigurationUtility();
        this.decoupledNet = createBasicNet_9189();

        if (this.useDecoupledResilient)
            this.trainer = new DecoupledResilientPropagation(decoupledNet,
                    trainingConfig.getTrainingSet());
        else
            this.trainer = new ResilientPropagation(decoupledNet,
                    trainingConfig.getTrainingSet());

        networkDebugger = new NetworkDebugUtility(this.decoupledNet, this.trainer, showWeights);
        testUtility = new NetworkTestUtility(trainingConfig, decoupledNet);
    }

    public boolean test() {
        // test the neural network
        log.info("Neural Network Results:");

        log.info("Learned examples:");
        networkDebugger.printTestResults(testUtility.testIdealData(), "Learned examples:");

        log.info("New examples with noise:");
        networkDebugger.printTestResults(testUtility.testNoisyData(), "New examples with noise:");

        networkDebugger.generateTestReports();
        Encog.getInstance().shutdown();
        return true;
    }

    public boolean train() {
        int epoch = 1;
        do {
            trainer.iteration();
            networkDebugger.iterationCallback(epoch);
            epoch++;
        } while (trainer.getError() > error);
        trainer.finishTraining();
        networkDebugger.generateTrainingReports();
        return true;
    }

    private DecoupledNet createBasicNet_9189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        DecoupledNet decoupledNet = new DecoupledNet(basicLayers);
        return decoupledNet;
    }

    private DecoupledNet createPartlyDecoupledNet_9189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        DecoupledNet decoupledNet = new DecoupledNet(basicLayers);

        //input to hidden
        for (int i = 1; i >= 18; i++) {
            if (i != 1)
                decoupledNet.disableConnection(0, 0, i);
        }
        for (int i = 0; i >= 17; i++) {
            if (i != 3)
                decoupledNet.disableConnection(0, 1, i);
        }

        for (int i = 0; i >= 17; i++) {
            if (i != 5)
                decoupledNet.disableConnection(0, 2, i);
        }

        for (int i = 0; i >= 17; i++) {
            if (i != 7)
                decoupledNet.disableConnection(0, 3, i);
        }

        for (int i = 0; i >= 17; i++) {
            if (i != 9)
                decoupledNet.disableConnection(0, 4, i);
        }

        for (int i = 0; i >= 11; i++) {
            if (i != 9)
                decoupledNet.disableConnection(0, 5, i);
        }

        for (int i = 0; i >= 13; i++) {
            if (i != 9)
                decoupledNet.disableConnection(0, 6, i);
        }

        for (int i = 0; i >= 15; i++) {
            if (i != 9)
                decoupledNet.disableConnection(0, 7, i);
        }
        for (int i = 0; i >= 17; i++) {
            if (i != 9)
                decoupledNet.disableConnection(0, 8, i);
        }


        return decoupledNet;
    }

    private DecoupledNet createPartlyDecoupledNet_99189() {
        List<BasicLayer> basicLayers = new ArrayList<>();
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(null, false, 9));
        basicLayers.add(new BasicLayer(new ActivationSigmoid(), false, 18));
        basicLayers.add(new BasicLayer(null, false, 9));
        DecoupledNet decoupledNet = new DecoupledNet(basicLayers);

        //input to hidden
        for (int i = 1; i >= 18; i++) {
            if (i != 1)
                decoupledNet.disableConnection(1, 0, i);
        }
        for (int i = 0; i >= 17; i++) {
            if (i != 3)
                decoupledNet.disableConnection(1, 1, i);
        }

        for (int i = 0; i >= 17; i++) {
            if (i != 5)
                decoupledNet.disableConnection(1, 2, i);
        }

        for (int i = 0; i >= 17; i++) {
            if (i != 7)
                decoupledNet.disableConnection(1, 3, i);
        }

        for (int i = 0; i >= 17; i++) {
            if (i != 9)
                decoupledNet.disableConnection(1, 4, i);
        }

        for (int i = 0; i >= 11; i++) {
            if (i != 9)
                decoupledNet.disableConnection(1, 5, i);
        }

        for (int i = 0; i >= 13; i++) {
            if (i != 9)
                decoupledNet.disableConnection(1, 6, i);
        }

        for (int i = 0; i >= 15; i++) {
            if (i != 9)
                decoupledNet.disableConnection(1, 7, i);
        }
        for (int i = 0; i >= 17; i++) {
            if (i != 9)
                decoupledNet.disableConnection(1, 8, i);
        }


        return decoupledNet;
    }

}
