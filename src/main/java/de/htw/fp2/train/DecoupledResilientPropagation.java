package de.htw.fp2.train;

import org.encog.mathutil.EncogMath;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.ContainsFlat;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

/**
 * Created by vs on 2.01.15.
 */
public class DecoupledResilientPropagation extends ResilientPropagation {

    private double maxStep;
    private double[] lastWeightChange;

    public DecoupledResilientPropagation(ContainsFlat network, MLDataSet training) {
        super(network, training);
        this.maxStep = 50.0D;
        this.lastWeightChange = new double[network.getFlat().getWeights().length];
    }

    public DecoupledResilientPropagation(ContainsFlat network, MLDataSet training, double initialUpdate, double maxStep) {
        super(network, training, initialUpdate, maxStep);
        this.maxStep = maxStep;
        this.lastWeightChange = new double[network.getFlat().getWeights().length];
    }

    @Override
    public double updateWeightPlus(double[] gradients, double[] lastGradient, int index) {
        int change = EncogMath.sign(gradients[index] * lastGradient[index]);
        double delta;
        double weightChange = 0.0D;
        if (lastGradient[index] != 0) {
            if (change > 0) {
                delta = getUpdateValues()[index] * 1.2D;
                delta = Math.min(delta, this.maxStep);
                weightChange = (double) EncogMath.sign(gradients[index]) * delta;
                getUpdateValues()[index] = delta;
                lastGradient[index] = gradients[index];
            } else if (change < 0) {
                delta = getUpdateValues()[index] * 0.5D;
                delta = Math.max(delta, 1.0E-6D);
                getUpdateValues()[index] = delta;
                weightChange = -this.lastWeightChange[index];
                lastGradient[index] = 0.0D;
            } else if (change == 0) {
                delta = getUpdateValues()[index];
                weightChange = (double) EncogMath.sign(gradients[index]) * delta;
                lastGradient[index] = gradients[index];
            }
        }
        this.lastWeightChange[index] = weightChange;
        return weightChange;
    }
}
