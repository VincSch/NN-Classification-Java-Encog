package de.htw.fp2.dataset;

import de.htw.fp2.util.ArrayUtil;

import java.io.Serializable;
import java.util.Random;
import java.util.StringTokenizer;

/**
 * Created by patrick on 09.12.14.
 */
public class Pattern implements Serializable {

    private Random rand;
    public boolean[][] value = {
        { false, false, false },
        { false, false, false },
        { false, false, false }
    };

    public Pattern() {
        this.rand = new Random();
        this.rand.setSeed(System.currentTimeMillis());
    }

    public Pattern(boolean[][] pattern) {
        super();
        this.value = pattern;
    }

    public void addNoise(int range) {
        for (int i = 0; i < range; i++) {
            this.addNoise();
        }
    }

    public void addNoise() {
        int column = this.rand.nextInt(3);
        int row = this.rand.nextInt(3);
        this.value[row][column] = !this.value[row][column];
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        //todo validate array
        return true;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int row = 0; row < this.value.length; row++) {
            sb.append("[");
            for (int column = 0; column < this.value[row].length; column++) {
                sb.append(this.value[row][column] ? "1" : "0");
                if (column != this.value[row].length - 1)
                    sb.append(",");
            }
            sb.append("]");
            if (row != this.value.length - 1)
                sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }

    public static Pattern resolvePattern(String pattern) {
        boolean[][] value = new boolean[3][3];
        StringTokenizer st = new StringTokenizer(pattern, ",");
        return new Pattern(value);
    }

    public double[][] getDoubleValue() {
        double[][] doubleValue = new double[this.value.length][this.value[0].length];
        for (int i = 0; i < doubleValue.length; i++) {
            for (int j = 0; j < doubleValue[0].length; j++) {
                doubleValue[i][j] = this.value[i][j] ? 1.0 : 0.0;
            }
        }
        return doubleValue;
    }

    public double[] getFlat() {
        return ArrayUtil.flat(this.getDoubleValue());
    }
}
