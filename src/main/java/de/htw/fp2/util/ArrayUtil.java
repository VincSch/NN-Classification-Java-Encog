package de.htw.fp2.util;

/**
 * Created by patrick on 16.12.14.
 */
public class ArrayUtil {
    public static double[] flat(double[][] matrix) {
        double[] flat = new double[matrix.length * matrix[0].length];
        int counter = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                flat[counter] = matrix[i][j];
                counter++;
            }
        }
        return flat;
    }

    public static int[] flat(int[][] matrix) {
        int[] flat = new int[matrix.length * matrix[0].length];
        int counter = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                flat[counter] = matrix[i][j];
                counter++;
            }
        }
        return flat;
    }

    public static int[][] toMatrix(int[] flat, int rowCount) {
        int rows = flat.length / rowCount;
        int coloumns = flat.length / rows;
        int[][] matrix = new int[rows][coloumns];
        int counter = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < coloumns; j++) {
                matrix[i][j] = flat[counter];
                counter++;
            }
        }
        return matrix;
    }
}
