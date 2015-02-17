package de.htw.fp2.network.classification;

import de.htw.fp2.common.NetworkTestUtility;
import de.htw.fp2.train.DecoupledResilientPropagation;
import de.htw.fp2.train.common.TrainingConfigurationUtility;
import de.htw.fp2.util.NetworkDebugUtility;
import de.htw.fp2.network.DecoupledNet;
import org.apache.log4j.Logger;
import org.encog.Encog;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

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
    private static int MAX_EPOCHS = 2000;

    public DecoupledNetClassification(boolean useDecoupledResilient, DecoupledNet.Topology topology) {
        this.useDecoupledResilient = useDecoupledResilient;
        this.error = 0.01;
        trainingConfig = new TrainingConfigurationUtility();
        this.decoupledNet = new DecoupledNet(topology);

        if (this.useDecoupledResilient)
            this.trainer = new DecoupledResilientPropagation(decoupledNet,
                    trainingConfig.getTrainingSet());
        else
            this.trainer = new ResilientPropagation(decoupledNet,
                    trainingConfig.getTrainingSet());

        networkDebugger = new NetworkDebugUtility(this.decoupledNet, this.trainer, decoupledNet.getTopology());
        testUtility = new NetworkTestUtility(trainingConfig, decoupledNet);
    }

    public DecoupledNetClassification(boolean useDecoupledResilient, DecoupledNet.Topology topology, double error, int max_epoch) {
        this.useDecoupledResilient = useDecoupledResilient;
        this.error = error;
        MAX_EPOCHS = max_epoch;
        trainingConfig = new TrainingConfigurationUtility();
        this.decoupledNet = new DecoupledNet(topology);

        if (this.useDecoupledResilient)
            this.trainer = new DecoupledResilientPropagation(decoupledNet,
                    trainingConfig.getTrainingSet());
        else
            this.trainer = new ResilientPropagation(decoupledNet,
                    trainingConfig.getTrainingSet());

        networkDebugger = new NetworkDebugUtility(this.decoupledNet, this.trainer, decoupledNet.getTopology());
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
            decoupledNet.resetDisabledConnections();
            networkDebugger.iterationCallback(epoch);
            epoch++;
        } while (trainer.getError() > error && epoch < MAX_EPOCHS);
        trainer.finishTraining();
        networkDebugger.generateTrainingReports();
        return true;
    }

}
